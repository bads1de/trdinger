"""
自動戦略生成サービス

GA実行、進捗管理、結果保存などの統合機能を提供します。
"""

import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable
import logging

from ..models.strategy_gene import StrategyGene
from ..models.ga_config import GAConfig, GAProgress
from ..engines.ga_engine import GeneticAlgorithmEngine
from ..factories.strategy_factory import StrategyFactory
from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)
from database.repositories.backtest_result_repository import BacktestResultRepository
from database.connection import SessionLocal

logger = logging.getLogger(__name__)


class AutoStrategyService:
    """
    自動戦略生成サービス

    GA実行、進捗管理、結果保存を統合的に管理します。
    """

    def __init__(self):
        """初期化"""
        self.running_experiments: Dict[str, Dict[str, Any]] = {}
        self.progress_data: Dict[str, GAProgress] = {}

        # サービスの初期化
        self.strategy_factory = StrategyFactory()
        self.backtest_service = None
        self.ga_engine = None

        # リポジトリの初期化
        self.ga_experiment_repo = None
        self.generated_strategy_repo = None

        self._init_services()

    def _init_services(self):
        """サービスの初期化"""
        try:
            # データベースセッション
            db = SessionLocal()
            try:
                # リポジトリの初期化
                ohlcv_repo = OHLCVRepository(db)
                oi_repo = OpenInterestRepository(db)
                fr_repo = FundingRateRepository(db)
                self.ga_experiment_repo = GAExperimentRepository(db)
                self.generated_strategy_repo = GeneratedStrategyRepository(db)

                # サービスの初期化（拡張されたBacktestDataService）
                data_service = BacktestDataService(
                    ohlcv_repo=ohlcv_repo, oi_repo=oi_repo, fr_repo=fr_repo
                )
                self.backtest_service = BacktestService(data_service)

                # GAエンジンの初期化
                self.ga_engine = GeneticAlgorithmEngine(
                    self.backtest_service, self.strategy_factory
                )

                logger.info("自動戦略生成サービス初期化完了（OI/FR統合版）")

            finally:
                db.close()

        except Exception as e:
            logger.error(f"サービス初期化エラー: {e}")
            raise

    def start_strategy_generation(
        self,
        experiment_name: str,
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
        progress_callback: Optional[Callable[[GAProgress], None]] = None,
    ) -> str:
        """
        戦略生成を開始

        Args:
            experiment_name: 実験名
            ga_config: GA設定
            backtest_config: バックテスト設定
            progress_callback: 進捗コールバック

        Returns:
            実験ID
        """
        try:
            # 実験IDの生成
            experiment_id = str(uuid.uuid4())

            # 設定の検証
            is_valid, errors = ga_config.validate()
            if not is_valid:
                raise ValueError(f"Invalid GA config: {', '.join(errors)}")

            # データベースに実験を保存
            db = SessionLocal()
            try:
                ga_experiment_repo = GAExperimentRepository(db)
                db_experiment = ga_experiment_repo.create_experiment(
                    name=experiment_name,
                    config=ga_config.to_dict(),
                    total_generations=ga_config.generations,
                    status="starting",
                )
                db_experiment_id = db_experiment.id
            finally:
                db.close()

            # 実験情報を記録
            experiment_info = {
                "id": experiment_id,
                "db_id": db_experiment_id,
                "name": experiment_name,
                "ga_config": ga_config,
                "backtest_config": backtest_config,
                "status": "starting",
                "start_time": time.time(),
                "thread": None,
            }

            self.running_experiments[experiment_id] = experiment_info

            # 進捗コールバックの設定
            def combined_callback(progress: GAProgress):
                self.progress_data[experiment_id] = progress

                # データベースに進捗を保存
                try:
                    db = SessionLocal()
                    try:
                        ga_experiment_repo = GAExperimentRepository(db)
                        ga_experiment_repo.update_experiment_progress(
                            experiment_id=db_experiment_id,
                            current_generation=progress.current_generation,
                            progress=progress.progress_percentage / 100.0,
                            best_fitness=progress.best_fitness,
                        )
                    finally:
                        db.close()
                except Exception as e:
                    logger.error(f"進捗保存エラー: {e}")

                if progress_callback:
                    progress_callback(progress)

            self.ga_engine.set_progress_callback(combined_callback)

            # バックグラウンドで実行
            thread = threading.Thread(
                target=self._run_experiment,
                args=(experiment_id, ga_config, backtest_config),
                daemon=True,
            )

            experiment_info["thread"] = thread
            experiment_info["status"] = "running"

            thread.start()

            logger.info(f"戦略生成実験開始: {experiment_id} ({experiment_name})")
            return experiment_id

        except Exception as e:
            logger.error(f"戦略生成開始エラー: {e}")
            raise

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
            experiment_info = self.running_experiments[experiment_id]

            # バックテスト設定に実験IDを追加
            backtest_config["experiment_id"] = experiment_id

            # GA実行
            logger.info(f"GA実行開始: {experiment_id}")
            result = self.ga_engine.run_evolution(ga_config, backtest_config)

            # 結果の保存
            self._save_experiment_result(
                experiment_id, result, ga_config, backtest_config
            )

            # 実験完了
            experiment_info["status"] = "completed"
            experiment_info["end_time"] = time.time()
            experiment_info["result"] = result

            # データベースで実験を完了状態にする
            try:
                db = SessionLocal()
                try:
                    ga_experiment_repo = GAExperimentRepository(db)
                    ga_experiment_repo.complete_experiment(
                        experiment_id=experiment_info["db_id"],
                        best_fitness=result["best_fitness"],
                        final_generation=ga_config.generations,
                    )
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"実験完了状態更新エラー: {e}")

            # 最終進捗更新
            final_progress = GAProgress(
                experiment_id=experiment_id,
                current_generation=ga_config.generations,
                total_generations=ga_config.generations,
                best_fitness=result["best_fitness"],
                average_fitness=result["best_fitness"],  # 簡略化
                execution_time=result["execution_time"],
                estimated_remaining_time=0.0,
                status="completed",
            )

            self.progress_data[experiment_id] = final_progress

            logger.info(f"GA実行完了: {experiment_id}")

        except Exception as e:
            logger.error(f"実験実行エラー ({experiment_id}): {e}")

            # エラー状態の記録
            if experiment_id in self.running_experiments:
                self.running_experiments[experiment_id]["status"] = "error"
                self.running_experiments[experiment_id]["error"] = str(e)

                # データベースでエラー状態を更新
                try:
                    db = SessionLocal()
                    try:
                        ga_experiment_repo = GAExperimentRepository(db)
                        ga_experiment_repo.update_experiment_status(
                            experiment_id=self.running_experiments[experiment_id][
                                "db_id"
                            ],
                            status="error",
                        )
                    finally:
                        db.close()
                except Exception as db_error:
                    logger.error(f"エラー状態DB更新エラー: {db_error}")

            # エラー進捗更新
            error_progress = GAProgress(
                experiment_id=experiment_id,
                current_generation=0,
                total_generations=ga_config.generations,
                best_fitness=0.0,
                average_fitness=0.0,
                execution_time=0.0,
                estimated_remaining_time=0.0,
                status="error",
            )

            self.progress_data[experiment_id] = error_progress

    def get_experiment_progress(self, experiment_id: str) -> Optional[GAProgress]:
        """
        実験の進捗を取得

        Args:
            experiment_id: 実験ID

        Returns:
            進捗情報（存在しない場合はNone）
        """
        return self.progress_data.get(experiment_id)

    def get_experiment_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        実験結果を取得

        Args:
            experiment_id: 実験ID

        Returns:
            実験結果（存在しない場合はNone）
        """
        experiment_info = self.running_experiments.get(experiment_id)
        if experiment_info and experiment_info["status"] == "completed":
            return experiment_info.get("result")
        return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        実験一覧を取得

        Returns:
            実験情報のリスト
        """
        experiments = []
        for experiment_id, info in self.running_experiments.items():
            experiment_summary = {
                "id": experiment_id,
                "name": info["name"],
                "status": info["status"],
                "start_time": info["start_time"],
                "end_time": info.get("end_time"),
                "error": info.get("error"),
            }
            experiments.append(experiment_summary)

        return experiments

    def stop_experiment(self, experiment_id: str) -> bool:
        """
        実験を停止

        Args:
            experiment_id: 実験ID

        Returns:
            停止成功の場合True
        """
        try:
            experiment_info = self.running_experiments.get(experiment_id)
            if not experiment_info:
                return False

            if experiment_info["status"] == "running":
                # GA実行を停止
                self.ga_engine.stop_evolution()

                # 実験状態を更新
                experiment_info["status"] = "stopped"
                experiment_info["end_time"] = time.time()

                logger.info(f"実験停止: {experiment_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"実験停止エラー: {e}")
            return False

    def _save_experiment_result(
        self,
        experiment_id: str,
        result: Dict[str, Any],
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ):
        """
        実験結果をデータベースに保存

        Args:
            experiment_id: 実験ID
            result: GA実行結果
            ga_config: GA設定
            backtest_config: バックテスト設定
        """
        try:
            experiment_info = self.running_experiments.get(experiment_id)
            if not experiment_info:
                logger.error(f"実験情報が見つかりません: {experiment_id}")
                return

            db_experiment_id = experiment_info["db_id"]
            best_strategy = result["best_strategy"]
            best_fitness = result["best_fitness"]
            all_strategies = result.get("all_strategies", [])

            logger.info(f"実験結果保存開始: {experiment_id}")
            logger.info(f"最高フィットネス: {best_fitness:.4f}")
            logger.info(f"最良戦略ID: {best_strategy.id}")
            logger.info(f"使用指標数: {len(best_strategy.indicators)}")
            logger.info(f"保存対象戦略数: {len(all_strategies)}")

            # データベースに戦略を保存
            db = SessionLocal()
            try:
                generated_strategy_repo = GeneratedStrategyRepository(db)
                backtest_result_repo = BacktestResultRepository(db)

                # 最良戦略を保存
                best_strategy_record = generated_strategy_repo.save_strategy(
                    experiment_id=db_experiment_id,
                    gene_data=best_strategy.to_dict(),
                    generation=ga_config.generations,
                    fitness_score=best_fitness,
                )

                logger.info(f"最良戦略を保存しました: DB ID {best_strategy_record.id}")

                # 最良戦略のバックテスト結果をbacktest_resultsテーブルにも保存
                try:
                    logger.info("最良戦略のバックテスト結果を詳細保存開始...")

                    # 最良戦略のバックテストを再実行（詳細データ取得のため）
                    detailed_backtest_config = backtest_config.copy()
                    detailed_backtest_config["strategy_name"] = (
                        f"AUTO_STRATEGY_{experiment_info['name']}_{best_strategy.id[:8]}"
                    )
                    detailed_backtest_config["strategy_config"] = {
                        "strategy_type": "GENERATED_AUTO",
                        "parameters": {"strategy_gene": best_strategy.to_dict()},
                    }

                    # バックテスト実行
                    detailed_result = self.backtest_service.run_backtest(
                        detailed_backtest_config
                    )

                    # backtest_resultsテーブル用のデータ構造を構築
                    backtest_result_data = {
                        "strategy_name": detailed_backtest_config["strategy_name"],
                        "symbol": detailed_backtest_config["symbol"],
                        "timeframe": detailed_backtest_config["timeframe"],
                        "start_date": detailed_backtest_config["start_date"],
                        "end_date": detailed_backtest_config["end_date"],
                        "initial_capital": detailed_backtest_config["initial_capital"],
                        "commission_rate": detailed_backtest_config.get(
                            "commission_rate", 0.001
                        ),
                        "config_json": {
                            "strategy_config": detailed_backtest_config[
                                "strategy_config"
                            ],
                            "experiment_id": experiment_id,
                            "db_experiment_id": db_experiment_id,
                            "fitness_score": best_fitness,
                        },
                        "performance_metrics": detailed_result.get(
                            "performance_metrics", {}
                        ),
                        "equity_curve": detailed_result.get("equity_curve", []),
                        "trade_history": detailed_result.get("trade_history", []),
                        "execution_time": detailed_result.get("execution_time"),
                        "status": "completed",
                    }

                    # backtest_resultsテーブルに保存
                    saved_backtest_result = backtest_result_repo.save_backtest_result(
                        backtest_result_data
                    )
                    logger.info(
                        f"最良戦略のバックテスト結果を保存しました: ID {saved_backtest_result.get('id')}"
                    )

                except Exception as e:
                    logger.error(f"最良戦略のバックテスト結果保存エラー: {e}")
                    # エラーが発生してもメイン処理は継続

                # 全戦略を一括保存（オプション）
                if all_strategies and len(all_strategies) > 1:
                    strategies_data = []
                    for i, strategy in enumerate(
                        all_strategies[:100]
                    ):  # 最大100戦略まで保存
                        if strategy != best_strategy:  # 最良戦略は既に保存済み
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
                        saved_strategies = (
                            generated_strategy_repo.save_strategies_batch(
                                strategies_data
                            )
                        )
                        logger.info(
                            f"追加戦略を一括保存しました: {len(saved_strategies)} 件"
                        )

            finally:
                db.close()

            logger.info(f"実験結果保存完了: {experiment_id}")

        except Exception as e:
            logger.error(f"実験結果保存エラー: {e}")
            raise

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
            logger.error(f"戦略テスト実行エラー: {e}")
            return {"success": False, "error": str(e)}
