"""
実験管理マネージャー

GA実験の実行と管理を担当します。
"""

import logging
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

from app.services.backtest.services.backtest_service import BacktestService

from ..config.ga import GAConfig
from ..core.engine.evolution_runner import EvolutionStoppedError
from .experiment_backtest_service import ExperimentBacktestService
from .experiment_persistence_service import ExperimentPersistenceService

if TYPE_CHECKING:
    from ..core.ga_engine import GeneticAlgorithmEngine

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    実験管理マネージャー

    GA実験の実行と管理を担当します。
    """

    _active_engines: ClassVar[Dict[str, "GeneticAlgorithmEngine"]] = {}
    _registry_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(
        self,
        backtest_service: BacktestService,
        persistence_service: ExperimentPersistenceService,
    ):
        """初期化"""
        self.backtest_service = backtest_service
        self.persistence_service = persistence_service
        self.experiment_backtest_service = ExperimentBacktestService(backtest_service)

    def run_experiment(
        self, experiment_id: str, ga_config: GAConfig, backtest_config: Dict[str, Any]
    ):
        """
        GA実験を非同期（バックグラウンド）実行の文脈で処理

        GAエンジンの実行、結果の永続化、およびステータス管理（完了/失敗）を
        一連のフローとして実行します。

        Args:
            experiment_id: 実験の一意識別子（UUID）
            ga_config: GA実行設定
            backtest_config: バックテスト実行設定
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context=f"GA実験実行 ({experiment_id})", is_api_call=False)
        def _execute():
            engine = self._get_active_engine(experiment_id)
            if not engine:
                raise RuntimeError(
                    f"GAエンジンが初期化されていません: {experiment_id}"
                )

            run_backtest_config = backtest_config.copy()
            run_backtest_config["experiment_id"] = experiment_id

            try:
                logger.info(f"GA実行開始: {experiment_id}")

                # GA実行
                result = engine.run_evolution(ga_config, run_backtest_config)

                if engine.is_stop_requested() is True:
                    logger.info(
                        f"GA実行は停止要求により中断されました: {experiment_id}"
                    )
                    self.persistence_service.stop_experiment(experiment_id)
                    return

                experiment_info = self.persistence_service.get_experiment_info(
                    experiment_id
                )

                detailed_backtest_result_data = (
                    self.experiment_backtest_service.create_detailed_backtest_result_data(
                        result=result,
                        ga_config=ga_config,
                        backtest_config=run_backtest_config,
                        experiment_id=experiment_id,
                        experiment_info=experiment_info,
                    )
                )

                # 結果の永続化
                self.persistence_service.save_experiment_result(
                    experiment_id,
                    result,
                    ga_config,
                    run_backtest_config,
                    experiment_info=experiment_info,
                )

                self.persistence_service.save_backtest_result(
                    detailed_backtest_result_data
                )

                if engine.is_stop_requested() is True:
                    logger.info(
                        f"GA実行は結果保存後に停止要求を検知しました: {experiment_id}"
                    )
                    self.persistence_service.stop_experiment(experiment_id)
                    return

                self.persistence_service.complete_experiment(experiment_id)

                logger.info(f"GA実行完了: {experiment_id}")

            except EvolutionStoppedError:
                logger.info(f"GA実行停止を検知しました: {experiment_id}")
                self.persistence_service.stop_experiment(experiment_id)
            finally:
                self.release_experiment(experiment_id, engine)

        try:
            _execute()
        except Exception as e:
            logger.error(f"実験実行エラー ({experiment_id}): {e}")
            self.persistence_service.fail_experiment(experiment_id)
            self.release_experiment(experiment_id)

    def initialize_ga_engine(
        self, ga_config: GAConfig, experiment_id: Optional[str] = None
    ) -> "GeneticAlgorithmEngine":
        """GAエンジンを初期化（Factoryを使用）"""
        from ..core.ga_engine_factory import GeneticAlgorithmEngineFactory

        engine = GeneticAlgorithmEngineFactory.create_engine(
            self.backtest_service, ga_config
        )
        if experiment_id:
            self._register_active_engine(experiment_id, engine)

        return engine

    def stop_experiment(self, experiment_id: str) -> bool:
        """
        実行中の実験に停止シグナルを送信

        Args:
            experiment_id: 停止対象の実験ID

        Returns:
            停止処理が受け付けられた場合はTrue
        """
        engine = self._get_active_engine(experiment_id)

        if engine:
            engine.stop_evolution()
            self.persistence_service.stop_experiment(experiment_id)
            logger.info(f"実験停止シグナル送信: {experiment_id}")
            return True

        experiment_info = self.persistence_service.get_experiment_info(experiment_id)
        if experiment_info and experiment_info.get("status") == "running":
            self.persistence_service.stop_experiment(experiment_id)
            logger.info(
                f"実行中の実験を停止状態に更新しました（実行コンテキストは未検出）: {experiment_id}"
            )
            return True

        logger.warning(f"停止対象の実行中実験が見つかりません: {experiment_id}")
        return False

    def release_experiment(
        self,
        experiment_id: str,
        engine: Optional["GeneticAlgorithmEngine"] = None,
    ) -> None:
        """実験の実行コンテキストをレジストリから解放します。"""
        self._release_active_engine(experiment_id, engine)

    @classmethod
    def _register_active_engine(
        cls, experiment_id: str, engine: "GeneticAlgorithmEngine"
    ) -> None:
        with cls._registry_lock:
            cls._active_engines[experiment_id] = engine

    @classmethod
    def _get_active_engine(
        cls, experiment_id: str
    ) -> Optional["GeneticAlgorithmEngine"]:
        with cls._registry_lock:
            return cls._active_engines.get(experiment_id)

    @classmethod
    def _release_active_engine(
        cls,
        experiment_id: str,
        engine: Optional["GeneticAlgorithmEngine"] = None,
    ) -> None:
        with cls._registry_lock:
            current = cls._active_engines.get(experiment_id)
            if current is None:
                return
            if engine is None or current is engine:
                cls._active_engines.pop(experiment_id, None)
