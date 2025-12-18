"""
実験管理マネージャー

GA実験の実行と管理を担当します。
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from app.services.backtest.backtest_service import BacktestService

from ..config.ga import GAConfig
from .experiment_persistence_service import ExperimentPersistenceService

if TYPE_CHECKING:
    from ..core.ga_engine import GeneticAlgorithmEngine
    from ..generators.random_gene_generator import RandomGeneGenerator

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
        self.ga_engine: Optional["GeneticAlgorithmEngine"] = None

    def run_experiment(
        self, experiment_id: str, ga_config: GAConfig, backtest_config: Dict[str, Any]
    ):
        """
        実験をバックグラウンドで実行
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context=f"GA実験実行 ({experiment_id})", is_api_call=False)
        def _execute():
            if not self.ga_engine:
                raise RuntimeError("GAエンジンが初期化されていません。")

            logger.info(f"GA実行開始: {experiment_id}")
            
            # コンテキスト情報の付与
            backtest_config["experiment_id"] = experiment_id

            # GA実行
            result = self.ga_engine.run_evolution(ga_config, backtest_config)

            # 結果の永続化
            self.persistence_service.save_experiment_result(experiment_id, result, ga_config, backtest_config)
            self.persistence_service.complete_experiment(experiment_id)

            logger.info(f"GA実行完了: {experiment_id}")

        try:
            _execute()
        except Exception as e:
            logger.error(f"実験実行エラー ({experiment_id}): {e}")
            self.persistence_service.fail_experiment(experiment_id)

    def initialize_ga_engine(self, ga_config: GAConfig):
        """GAエンジンを初期化（Factoryを使用）"""
        from ..core.ga_engine_factory import GeneticAlgorithmEngineFactory
        
        self.ga_engine = GeneticAlgorithmEngineFactory.create_engine(
            self.backtest_service, ga_config
        )

    def stop_experiment(self, experiment_id: str) -> bool:
        """実験を停止"""
        if self.ga_engine:
            self.ga_engine.stop_evolution()

        self.persistence_service.stop_experiment(experiment_id)
        logger.info(f"実験停止シグナル送信: {experiment_id}")
        return True
