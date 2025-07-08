"""
実験管理マネージャー

GA実験の実行と管理を担当します。
"""

import logging
from typing import Dict, Any, List, Optional

from ..models.ga_config import GAConfig
from ..models.strategy_gene import StrategyGene
from ..engines.ga_engine import GeneticAlgorithmEngine
from ..factories.strategy_factory import StrategyFactory
from ..generators.random_gene_generator import RandomGeneGenerator
from app.core.services.backtest_service import BacktestService
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

        except Exception as e:
            logger.error(f"GA実験の実行中にエラーが発生しました ({experiment_id}): {e}")

            # 実験を失敗状態にする
            self.persistence_service.fail_experiment(experiment_id)

            # エラー進捗を作成・通知

    def initialize_ga_engine(self, ga_config: GAConfig):
        """GAエンジンを初期化"""
        gene_generator = RandomGeneGenerator(ga_config)
        self.ga_engine = GeneticAlgorithmEngine(
            self.backtest_service, self.strategy_factory, gene_generator
        )
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
            test_config["strategy_config"] = {
                "strategy_type": "GENERATED_TEST",
                "parameters": {"strategy_gene": gene.to_dict()},
            }

            result = self.backtest_service.run_backtest(test_config)

            return {
                "success": True,
                "strategy_gene": gene.to_dict(),
                "backtest_result": result,
            }

        except Exception as e:
            logger.error(f"戦略のテスト実行中にエラーが発生しました: {e}")
            return {"success": False, "error": str(e)}
