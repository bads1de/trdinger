"""
実験管理マネージャー

GA実験の実行と管理を担当します。
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.backtest.backtest_service import BacktestService
from database.connection import get_db
from database.repositories.backtest_result_repository import BacktestResultRepository

from ..engines.ga_engine import GeneticAlgorithmEngine
from ..factories.strategy_factory import StrategyFactory
from ..generators.random_gene_generator import RandomGeneGenerator
from ..models.ga_config import GAConfig
from ..models.gene_strategy import StrategyGene
from ..services.experiment_persistence_service import ExperimentPersistenceService

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    実験管理マネージャー

    GA実験の実行と管理を担当します。
    """

    def __init__(
        self,
        backtest_service: BacktestService,
        persistence_service: ExperimentPersistenceService,
    ):
        """初期化"""
        self.backtest_service = backtest_service
        self.persistence_service = persistence_service
        self.strategy_factory = StrategyFactory()
        self.ga_engine: Optional[GeneticAlgorithmEngine] = None

    def run_experiment(
        self, experiment_id: str, ga_config: GAConfig, backtest_config: Dict[str, Any]
    ):
        """
        実験をバックグラウンドで実行

        Args:
            experiment_id: 実験ID
            ga_config: GA設定
            backtest_config: バックテスト設定
        """
        try:
            # バックテスト設定に実験IDを追加
            backtest_config["experiment_id"] = experiment_id

            # GA実行
            logger.info(f"GA実行開始: {experiment_id}")
            if not self.ga_engine:
                raise RuntimeError("GAエンジンが初期化されていません。")
            result = self.ga_engine.run_evolution(ga_config, backtest_config)

            # 実験結果を保存
            self.persistence_service.save_experiment_result(
                experiment_id, result, ga_config, backtest_config
            )

            # 実験を完了状態にする
            self.persistence_service.complete_experiment(experiment_id)

            # 最終進捗を作成・通知

            logger.info(f"GA実行完了: {experiment_id}")

            # 取引数0の問題を分析
            self._analyze_zero_trades_issue(experiment_id, result)

        except Exception as e:
            logger.error(f"GA実験の実行中にエラーが発生しました ({experiment_id}): {e}")

            # 実験を失敗状態にする
            self.persistence_service.fail_experiment(experiment_id)

            # エラー進捗を作成・通知

    def initialize_ga_engine(self, ga_config: GAConfig):
        """GAエンジンを初期化"""
        # GAConfigのログレベルを適用
        auto_strategy_logger = logging.getLogger("app.services.auto_strategy")
        auto_strategy_logger.setLevel(getattr(logging, ga_config.log_level.upper()))

        gene_generator = RandomGeneGenerator(ga_config)
        self.ga_engine = GeneticAlgorithmEngine(
            self.backtest_service, self.strategy_factory, gene_generator
        )
        if ga_config.log_level.upper() in ["DEBUG", "INFO"]:
            logger.info("GAエンジンを動的に初期化しました。")

    def stop_experiment(self, experiment_id: str) -> bool:
        """実験を停止"""
        try:
            # GA実行を停止
            if self.ga_engine:
                self.ga_engine.stop_evolution()

            # 実験を停止状態にする
            # 永続化サービス経由でステータスを更新
            self.persistence_service.stop_experiment(experiment_id)
            logger.info(f"実験停止: {experiment_id}")
            return True

        except Exception as e:
            logger.error(f"GA実験の停止中にエラーが発生しました: {e}")
            return False

    def validate_strategy_gene(self, gene: StrategyGene) -> tuple[bool, List[str]]:
        """
        戦略遺伝子の妥当性を検証

        Args:
            gene: 検証する戦略遺伝子

        Returns:
            (is_valid, error_messages)
        """
        return self.strategy_factory.validate_gene(gene)

    def test_strategy_generation(
        self, gene: StrategyGene, backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        単一戦略のテスト生成・実行

        Args:
            gene: テストする戦略遺伝子
            backtest_config: バックテスト設定

        Returns:
            テスト結果
        """
        try:
            # 戦略の妥当性チェック
            is_valid, errors = self.validate_strategy_gene(gene)
            if not is_valid:
                return {"success": False, "errors": errors}

            # 戦略クラスを生成
            self.strategy_factory.create_strategy_class(gene)

            # バックテスト実行
            test_config = backtest_config.copy()
            test_config["strategy_name"] = f"TEST_{gene.id}"
            from app.services.auto_strategy.models.gene_serialization import (
                GeneSerializer,
            )

            serializer = GeneSerializer()
            test_config["strategy_config"] = {
                "strategy_type": "GENERATED_TEST",
                "parameters": {"strategy_gene": serializer.strategy_gene_to_dict(gene)},
            }

            result = self.backtest_service.run_backtest(test_config)

            return {
                "success": True,
                "strategy_gene": serializer.strategy_gene_to_dict(gene),
                "backtest_result": result,
            }

        except Exception as e:
            logger.error(f"戦略テスト実行エラー: {e}")
            return {"success": False, "error": str(e)}

    def _analyze_zero_trades_issue(self, experiment_id: str, result: Dict[str, Any]):
        """取引数0の問題を分析してログ出力"""
        try:
            best_strategy = result.get("best_strategy")
            if not best_strategy:
                logger.warning(f"実験 {experiment_id}: ベスト戦略が見つかりません")
                return

            # 最新のバックテスト結果を取得して分析
            # 最新のバックテスト結果を取得して分析
            db = next(get_db())
            try:
                backtest_repo = BacktestResultRepository(db)
                recent_results = backtest_repo.get_recent_backtest_results(limit=10)

                zero_trade_count = 0
                for backtest_result in recent_results:
                    metrics = backtest_result.get("performance_metrics")
                    if metrics:
                        total_trades = metrics.get("total_trades", 0)

                        if total_trades == 0:
                            zero_trade_count += 1
                            logger.warning(
                                f"🔍 取引数0の戦略発見 (ID: {backtest_result.get('id')})"
                            )

                            # 戦略遺伝子を分析
                            strategy_config = backtest_result.get("config_json")
                            if strategy_config:
                                strategy_gene_dict = strategy_config.get(
                                    "parameters", {}
                                ).get("strategy_gene", {})

                                result_id = backtest_result.get("id")
                                if strategy_gene_dict and result_id is not None:
                                    self._analyze_strategy_gene_for_zero_trades(
                                        strategy_gene_dict, str(result_id)
                                    )

                if zero_trade_count > 0:
                    logger.warning(
                        f"実験 {experiment_id}: 最近の結果で {zero_trade_count}/10 の戦略が取引数0でした"
                    )
                else:
                    logger.info(
                        f"実験 {experiment_id}: 最近の結果で取引数0の問題は見つかりませんでした"
                    )

            finally:
                db.close()

        except Exception as e:
            logger.warning(
                f"取引数0の分析処理中にエラーが発生しました: {e}", exc_info=True
            )

    def _analyze_strategy_gene_for_zero_trades(
        self, strategy_gene_dict: Dict[str, Any], result_id: str
    ):
        """戦略遺伝子を分析して取引数0の原因を特定"""
        try:
            logger.info(f"      📊 戦略分析 (結果ID: {result_id}):")

            # インジケーター分析
            indicators = strategy_gene_dict.get("indicators", [])

            for indicator in indicators:
                # 未使用変数の代入を避け、直接参照してログに出すことで解析の可視性を向上
                ind_type = indicator.get("type", "Unknown")
                params = indicator.get("parameters", {})
                logger.info(f"        インジケーター: type={ind_type}, params={params}")

            # エントリー条件分析
            entry_conditions = strategy_gene_dict.get("entry_conditions", [])
            long_entry_conditions = strategy_gene_dict.get("long_entry_conditions", [])
            short_entry_conditions = strategy_gene_dict.get(
                "short_entry_conditions", []
            )

            # 条件の詳細分析
            all_conditions = (
                entry_conditions + long_entry_conditions + short_entry_conditions
            )
            problematic_conditions = []

            for i, condition in enumerate(all_conditions):
                left = condition.get("left_operand", "")
                operator = condition.get("operator", "")
                right = condition.get("right_operand", "")

                logger.info(f"          条件{i+1}: {left} {operator} {right}")

                # 問題のある条件をチェック
                if left == "MACD" or right == "MACD":
                    problematic_conditions.append(f"MACD参照問題 (条件{i+1})")

                if (
                    isinstance(right, str)
                    and right.replace(".", "").replace("-", "").isdigit()
                ):
                    try:
                        num_value = float(right)
                        if left in ["RSI", "CCI", "STOCH"] and (
                            num_value < 0 or num_value > 100
                        ):
                            problematic_conditions.append(
                                f"範囲外の値 (条件{i+1}: {left} {operator} {right})"
                            )
                    except Exception:
                        pass

            # 問題の報告
            if problematic_conditions:
                logger.warning("        🚨 問題のある条件:")
                for problem in problematic_conditions:
                    logger.warning(f"          - {problem}")
            else:
                logger.info("        ✅ 条件に明らかな問題は見つかりませんでした")

            # エグジット条件分析
            exit_conditions = strategy_gene_dict.get("exit_conditions", [])
            logger.info(f"        エグジット条件数: {len(exit_conditions)}")

            # リスク管理分析
            risk_management = strategy_gene_dict.get("risk_management", {})
            logger.info(f"        リスク管理設定: {risk_management}")

        except Exception as e:
            logger.error(f"戦略遺伝子分析エラー: {e}")
