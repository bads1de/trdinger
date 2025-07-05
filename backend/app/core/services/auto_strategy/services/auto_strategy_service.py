"""
自動戦略生成サービス
GA実行、進捗管理、結果保存を統合的に管理します。
複雑な分離構造を削除し、直接的で理解しやすい実装に変更しました。
"""

import logging
import uuid
from typing import Dict, Any, List, Optional

from fastapi import BackgroundTasks
from ..models.ga_config import GAConfig, GAProgress
from ..models.strategy_gene import StrategyGene
from ..engines.ga_engine import GeneticAlgorithmEngine
from ..factories.strategy_factory import StrategyFactory
from ..generators.random_gene_generator import RandomGeneGenerator
from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService

# データベースリポジトリのインポート
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.backtest_result_repository import (
    BacktestResultRepository,
)
from database.connection import SessionLocal

logger = logging.getLogger(__name__)


class AutoStrategyService:
    """
    自動戦略生成サービス

    GA実行、進捗管理、結果保存を統合的に管理します。
    """

    def __init__(self):
        """初期化"""
        # データベースセッションファクトリ
        self.db_session_factory = SessionLocal

        # サービスの初期化
        self.strategy_factory = StrategyFactory()
        self.backtest_service: BacktestService
        self.ga_engine: Optional[GeneticAlgorithmEngine] = None

        # 進捗管理（統合版）
        self.progress_data: Dict[str, GAProgress] = {}

        self._init_services()

    def _init_services(self):
        """
        サービスの初期化

        必要最小限のサービス初期化を行います。
        複雑な依存関係管理を削除し、シンプルな実装に変更しました。
        """
        try:
            # データベースリポジトリの初期化
            with self.db_session_factory() as db:
                from database.repositories.ohlcv_repository import OHLCVRepository
                from database.repositories.open_interest_repository import (
                    OpenInterestRepository,
                )
                from database.repositories.funding_rate_repository import (
                    FundingRateRepository,
                )

                ohlcv_repo = OHLCVRepository(db)
                oi_repo = OpenInterestRepository(db)
                fr_repo = FundingRateRepository(db)

                # バックテストサービスの初期化
                data_service = BacktestDataService(
                    ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
                )
                self.backtest_service = BacktestService(data_service)

            # GAエンジンは実行時に動的に初期化
            self.ga_engine = None

            logger.info("自動戦略生成サービス初期化完了")

        except Exception as e:
            logger.error(f"AutoStrategyServiceの初期化中にエラーが発生しました: {e}")
            raise

    def start_strategy_generation(
        self,
        experiment_name: str,
        ga_config_dict: Dict[str, Any],
        backtest_config_dict: Dict[str, Any],
        background_tasks: BackgroundTasks,
    ) -> str:
        """
        戦略生成を開始

        Args:
            experiment_name: 実験名
            ga_config_dict: GA設定の辞書
            backtest_config_dict: バックテスト設定の辞書
            background_tasks: FastAPIのバックグラウンドタスク

        Returns:
            実験ID
        """
        logger.info(f"戦略生成開始: {experiment_name}")

        # 1. GA設定の構築と検証
        try:
            ga_config = GAConfig.from_dict(ga_config_dict)
            is_valid, errors = ga_config.validate()
            if not is_valid:
                raise ValueError(f"無効なGA設定です: {', '.join(errors)}")
        except Exception as e:
            logger.error(f"GA設定の構築または検証に失敗しました: {e}", exc_info=True)
            raise ValueError(f"GA設定の構築または検証に失敗しました: {e}")

        # 2. バックテスト設定のシンボル正規化
        backtest_config = backtest_config_dict.copy()
        original_symbol = backtest_config.get("symbol")
        if original_symbol and ":" not in original_symbol:
            normalized_symbol = f"{original_symbol}:USDT"
            backtest_config["symbol"] = normalized_symbol
            logger.info(
                f"シンボルを正規化しました: {original_symbol} -> {normalized_symbol}"
            )

        # 3. 実験を作成（統合版）
        experiment_id = self._create_experiment(
            experiment_name, ga_config, backtest_config
        )

        # 4. GAエンジンを初期化
        gene_generator = RandomGeneGenerator(ga_config)
        self.ga_engine = GeneticAlgorithmEngine(
            self.backtest_service, self.strategy_factory, gene_generator
        )
        logger.info("GAエンジンを動的に初期化しました。")

        # 5. 実験をバックグラウンドで開始
        background_tasks.add_task(
            self._run_experiment, experiment_id, ga_config, backtest_config
        )

        logger.info(
            f"戦略生成実験のバックグラウンドタスクを追加しました: {experiment_id}"
        )
        return experiment_id

    def _run_experiment(
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
            self._save_experiment_result(
                experiment_id, result, ga_config, backtest_config
            )

            # 実験を完了状態にする（統合版）
            self._complete_experiment(experiment_id, result)

            # 最終進捗を作成・通知（統合版）
            self._create_final_progress(experiment_id, result, ga_config)

            logger.info(f"GA実行完了: {experiment_id}")

        except Exception as e:
            logger.error(f"GA実験の実行中にエラーが発生しました ({experiment_id}): {e}")

            # 実験を失敗状態にする（統合版）
            self._fail_experiment(experiment_id, str(e))

            # エラー進捗を作成・通知（統合版）
            self._create_error_progress(experiment_id, ga_config, str(e))

    def get_experiment_progress(self, experiment_id: str) -> Optional[GAProgress]:
        """実験の進捗を取得（統合版）"""
        return self.progress_data.get(experiment_id)

    def get_experiment_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """実験結果を取得（統合版）"""
        return self._get_experiment_result(experiment_id)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """実験一覧を取得（統合版）"""
        return self._list_experiments()

    def stop_experiment(self, experiment_id: str) -> bool:
        """実験を停止"""
        try:
            # GA実行を停止
            if self.ga_engine:
                self.ga_engine.stop_evolution()

            # 実験を停止状態にする
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                db_experiment_id = int(experiment_id)
                ga_experiment_repo.update_experiment_status(db_experiment_id, "stopped")
            logger.info(f"実験停止: {experiment_id}")
            return True

        except Exception as e:
            logger.error(f"GA実験の停止中にエラーが発生しました: {e}")
            return False

    def _save_experiment_result(
        self,
        experiment_id: str,
        result: Dict[str, Any],
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ):
        """実験結果をデータベースに保存（統合版）"""
        try:
            # 実験情報を取得（統合版）
            experiment_info = self._get_experiment_info(experiment_id)
            if not experiment_info:
                logger.error(f"実験情報が見つかりません: {experiment_id}")
                return

            logger.info(f"実験結果保存開始: {experiment_id}")

            with self.db_session_factory() as db:
                # 最良戦略の保存と詳細バックテストの実行
                self._save_best_strategy_and_run_detailed_backtest(
                    db,
                    experiment_id,
                    experiment_info,
                    result,
                    ga_config,
                    backtest_config,
                )

                # その他の戦略をバッチ保存
                self._save_other_strategies(db, experiment_info, result, ga_config)

            logger.info(f"実験結果保存完了: {experiment_id}")

        except Exception as e:
            logger.error(
                f"GA実験結果の保存中にエラーが発生しました: {e}", exc_info=True
            )
            raise

    def _save_best_strategy_and_run_detailed_backtest(
        self,
        db,
        experiment_id: str,
        experiment_info: Dict[str, Any],
        result: Dict[str, Any],
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ):
        """最良戦略を保存し、詳細なバックテストを実行して結果を保存する"""
        generated_strategy_repo = GeneratedStrategyRepository(db)
        backtest_result_repo = BacktestResultRepository(db)

        db_experiment_id = experiment_info["db_id"]
        best_strategy = result["best_strategy"]
        best_fitness = result["best_fitness"]

        # 1. 最良戦略を generated_strategies に保存
        best_strategy_record = generated_strategy_repo.save_strategy(
            experiment_id=db_experiment_id,
            gene_data=best_strategy.to_dict(),
            generation=ga_config.generations,
            fitness_score=best_fitness,
        )
        logger.info(f"最良戦略を保存しました: DB ID {best_strategy_record.id}")

        # 2. 最良戦略の詳細バックテストを実行し、backtest_results に保存
        try:
            logger.info("最良戦略の詳細バックテストと結果保存を開始...")
            detailed_backtest_config = self._prepare_detailed_backtest_config(
                best_strategy, experiment_info, backtest_config
            )
            detailed_result = self.backtest_service.run_backtest(
                detailed_backtest_config
            )

            backtest_result_data = self._prepare_backtest_result_data(
                detailed_result,
                detailed_backtest_config,
                experiment_id,
                db_experiment_id,
                best_fitness,
            )
            saved_backtest_result = backtest_result_repo.save_backtest_result(
                backtest_result_data
            )
            logger.info(
                f"最良戦略のバックテスト結果を保存しました: ID {saved_backtest_result.get('id')}"
            )

        except Exception as e:
            logger.error(
                f"最良戦略の詳細バックテスト結果の保存中にエラー: {e}", exc_info=True
            )

    def _prepare_detailed_backtest_config(
        self,
        best_strategy: StrategyGene,
        experiment_info: Dict[str, Any],
        backtest_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """詳細バックテスト用の設定を準備する"""
        config = backtest_config.copy()
        config["strategy_name"] = (
            f"AUTO_STRATEGY_{experiment_info['name']}_{best_strategy.id[:8]}"
        )
        config["strategy_config"] = {
            "strategy_type": "GENERATED_AUTO",
            "parameters": {"strategy_gene": best_strategy.to_dict()},
        }
        return config

    def _prepare_backtest_result_data(
        self,
        detailed_result: Dict[str, Any],
        config: Dict[str, Any],
        experiment_id: str,
        db_experiment_id: int,
        best_fitness: float,
    ) -> Dict[str, Any]:
        """backtest_resultsテーブルに保存するためのデータを構築する"""
        return {
            "strategy_name": config["strategy_name"],
            "symbol": config["symbol"],
            "timeframe": config["timeframe"],
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "initial_capital": config["initial_capital"],
            "commission_rate": config.get("commission_rate", 0.001),
            "config_json": {
                "strategy_config": config["strategy_config"],
                "experiment_id": experiment_id,
                "db_experiment_id": db_experiment_id,
                "fitness_score": best_fitness,
            },
            "performance_metrics": detailed_result.get("performance_metrics", {}),
            "equity_curve": detailed_result.get("equity_curve", []),
            "trade_history": detailed_result.get("trade_history", []),
            "execution_time": detailed_result.get("execution_time"),
            "status": "completed",
        }

    def _save_other_strategies(
        self,
        db,
        experiment_info: Dict[str, Any],
        result: Dict[str, Any],
        ga_config: GAConfig,
    ):
        """最良戦略以外の戦略をバッチ保存する"""
        all_strategies = result.get("all_strategies", [])
        if not all_strategies or len(all_strategies) <= 1:
            return

        best_strategy = result["best_strategy"]
        db_experiment_id = experiment_info["db_id"]
        generated_strategy_repo = GeneratedStrategyRepository(db)

        strategies_data = []
        for i, strategy in enumerate(all_strategies[:100]):  # 上位100件に制限
            if strategy.id != best_strategy.id:
                strategies_data.append(
                    {
                        "experiment_id": db_experiment_id,
                        "gene_data": strategy.to_dict(),
                        "generation": ga_config.generations,
                        "fitness_score": result.get(
                            "fitness_scores", [0.0] * len(all_strategies)
                        )[i],
                    }
                )

        if strategies_data:
            saved_count = generated_strategy_repo.save_strategies_batch(strategies_data)
            logger.info(f"追加戦略を一括保存しました: {saved_count} 件")

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

    # 統合された機能メソッド

    def _create_experiment(
        self, experiment_name: str, ga_config: GAConfig, backtest_config: Dict[str, Any]
    ) -> str:
        """
        実験を作成（統合版）

        Args:
            experiment_name: 実験名
            ga_config: GA設定
            backtest_config: バックテスト設定

        Returns:
            実験ID
        """
        try:
            experiment_id = str(uuid.uuid4())

            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)

                # 設定データを準備
                config_data = {
                    "ga_config": ga_config.to_dict(),
                    "backtest_config": backtest_config,
                }

                # create_experimentメソッドの正しい引数で呼び出し
                db_experiment = ga_experiment_repo.create_experiment(
                    name=experiment_name,
                    config=config_data,
                    total_generations=ga_config.generations,
                    status="running",
                )
                logger.info(
                    f"実験を作成しました: {experiment_id} (DB ID: {db_experiment.id})"
                )

                # 実際のDB IDを返すように変更（実験情報取得で使用するため）
                return str(db_experiment.id)

        except Exception as e:
            logger.error(f"実験作成エラー: {e}")
            raise

    def _complete_experiment(self, experiment_id: str, result: Dict[str, Any]):
        """
        実験を完了状態にする（統合版）

        Args:
            experiment_id: 実験ID
            result: 実験結果
        """
        try:
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)

                # experiment_idはDBのIDなので、直接更新する
                db_experiment_id = int(experiment_id)
                ga_experiment_repo.update_experiment_status(
                    db_experiment_id, "completed"
                )

        except Exception as e:
            logger.error(f"実験完了処理エラー: {e}")

    def _fail_experiment(self, experiment_id: str, error_message: str):
        """
        実験を失敗状態にする（統合版）

        Args:
            experiment_id: 実験ID
            error_message: エラーメッセージ
        """
        try:
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)

                # experiment_idはDBのIDなので、直接更新する
                db_experiment_id = int(experiment_id)
                ga_experiment_repo.update_experiment_status(db_experiment_id, "failed")

        except Exception as e:
            logger.error(f"実験失敗処理エラー: {e}")

    def _create_final_progress(
        self, experiment_id: str, result: Dict[str, Any], ga_config: GAConfig
    ):
        """
        最終進捗を作成（統合版）

        Args:
            experiment_id: 実験ID
            result: 実験結果
            ga_config: GA設定
        """
        try:
            progress = GAProgress(
                experiment_id=experiment_id,
                current_generation=ga_config.generations,
                total_generations=ga_config.generations,
                best_fitness=result.get("best_fitness", 0.0),
                average_fitness=result.get("best_fitness", 0.0),
                execution_time=result.get("execution_time", 0.0),
                estimated_remaining_time=0.0,
                status="completed",
            )

            self.progress_data[experiment_id] = progress
            logger.info(f"最終進捗を作成しました: {experiment_id}")

        except Exception as e:
            logger.error(f"最終進捗作成エラー: {e}")

    def _create_error_progress(
        self, experiment_id: str, ga_config: GAConfig, error_message: str
    ):
        """
        エラー進捗を作成（統合版）

        Args:
            experiment_id: 実験ID
            ga_config: GA設定
            error_message: エラーメッセージ
        """
        try:
            progress = GAProgress(
                experiment_id=experiment_id,
                current_generation=0,
                total_generations=ga_config.generations,
                best_fitness=0.0,
                average_fitness=0.0,
                execution_time=0.0,
                estimated_remaining_time=0.0,
                status="error",
            )

            self.progress_data[experiment_id] = progress
            logger.info(f"エラー進捗を作成しました: {experiment_id}")

        except Exception as e:
            logger.error(f"エラー進捗作成エラー: {e}")

    def _get_experiment_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        実験結果を取得（統合版）

        Args:
            experiment_id: 実験ID

        Returns:
            実験結果
        """
        try:
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                experiments = ga_experiment_repo.get_experiments_by_status("completed")

                for exp in experiments:
                    if exp.experiment_name == experiment_id:  # 簡易的な実装
                        return {
                            "id": exp.id,
                            "experiment_name": exp.experiment_name,
                            "status": exp.status,
                            "created_at": (
                                exp.created_at.isoformat()
                                if exp.created_at is not None
                                else None
                            ),
                            "completed_at": (
                                exp.completed_at.isoformat()
                                if exp.completed_at is not None
                                else None
                            ),
                        }

                return None

        except Exception as e:
            logger.error(f"実験結果取得エラー: {e}")
            return None

    def _list_experiments(self) -> List[Dict[str, Any]]:
        """
        実験一覧を取得（統合版）

        Returns:
            実験一覧
        """
        try:
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)
                experiments = ga_experiment_repo.get_recent_experiments(limit=100)

                return [
                    {
                        "id": exp.id,
                        "experiment_name": exp.name,  # nameプロパティを使用
                        "status": exp.status,
                        "created_at": (
                            exp.created_at.isoformat()
                            if exp.created_at is not None
                            else None
                        ),
                        "completed_at": (
                            exp.completed_at.isoformat()
                            if exp.completed_at is not None
                            else None
                        ),
                    }
                    for exp in experiments
                ]

        except Exception as e:
            logger.error(f"実験一覧取得エラー: {e}")
            return []

    def _get_experiment_info(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        実験情報を取得（統合版）

        Args:
            experiment_id: 実験ID

        Returns:
            実験情報辞書
        """
        try:
            with self.db_session_factory() as db:
                ga_experiment_repo = GAExperimentRepository(db)

                # 実験IDで検索（改良版）
                experiments = ga_experiment_repo.get_recent_experiments(limit=100)
                for exp in experiments:
                    # 実験名またはDB IDがexperiment_idと一致するものを検索
                    # UUIDとして生成されたexperiment_idは実験名として保存されている
                    if (
                        hasattr(exp, "name")
                        and exp.name is not None
                        and experiment_id in exp.name
                    ) or str(exp.id) == experiment_id:
                        return {
                            "db_id": exp.id,
                            "name": exp.name,
                            "status": exp.status,
                            "config": exp.config,
                            "created_at": exp.created_at,
                            "completed_at": exp.completed_at,
                        }

                logger.warning(f"実験が見つかりません: {experiment_id}")
                return None

        except Exception as e:
            logger.error(f"実験情報取得エラー: {e}")
            return None
