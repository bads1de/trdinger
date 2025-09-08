"""
実験管理マネージャー

GA実験の実行と管理を担当します。
"""

import logging
from typing import Any, Dict, Optional

from app.services.backtest.backtest_service import BacktestService

from ..core.ga_engine import GeneticAlgorithmEngine
from ..generators.strategy_factory import StrategyFactory
from ..generators.random_gene_generator import RandomGeneGenerator
from ..config import GAConfig
from .experiment_persistence_service import ExperimentPersistenceService

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
        from app.utils.error_handler import safe_operation

        @safe_operation(context=f"GA実験実行 ({experiment_id})", is_api_call=False)
        def _run_experiment():
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

        try:
            _run_experiment()
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
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context=f"GA実験停止 ({experiment_id})",
            is_api_call=False,
            default_return=False,
        )
        def _stop_experiment():
            # GA実行を停止
            if self.ga_engine:
                self.ga_engine.stop_evolution()

            # 実験を停止状態にする
            # 永続化サービス経由でステータスを更新
            self.persistence_service.stop_experiment(experiment_id)
            logger.info(f"実験停止: {experiment_id}")
            return True

        return _stop_experiment()
