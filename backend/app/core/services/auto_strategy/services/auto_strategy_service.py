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

        self._init_services()

    def _init_services(self):
        """サービスの初期化"""
        try:
            # データベースセッション
            db = SessionLocal()
            try:
                ohlcv_repo = OHLCVRepository(db)
                data_service = BacktestDataService(ohlcv_repo)
                self.backtest_service = BacktestService(data_service)

                # GAエンジンの初期化
                self.ga_engine = GeneticAlgorithmEngine(
                    self.backtest_service, self.strategy_factory
                )

                logger.info("自動戦略生成サービス初期化完了")

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

            # 実験情報を記録
            experiment_info = {
                "id": experiment_id,
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
            # TODO: データベース保存の実装
            # 現在は簡略化してログ出力のみ

            best_strategy = result["best_strategy"]
            best_fitness = result["best_fitness"]

            logger.info(f"実験結果保存: {experiment_id}")
            logger.info(f"最高フィットネス: {best_fitness:.4f}")
            logger.info(f"最良戦略ID: {best_strategy.id}")
            logger.info(f"使用指標数: {len(best_strategy.indicators)}")

            # TODO: 以下を実装
            # 1. ga_experiments テーブルに実験情報を保存
            # 2. generated_strategies テーブルに生成された戦略を保存
            # 3. バックテスト結果の保存

        except Exception as e:
            logger.error(f"実験結果保存エラー: {e}")

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
            strategy_class = self.strategy_factory.create_strategy_class(gene)

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
