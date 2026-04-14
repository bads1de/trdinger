"""
実験開始・停止の application service。
"""

import logging
from typing import Any, Dict, Protocol

from ..config.ga import GAConfig

logger = logging.getLogger(__name__)


class TaskScheduler(Protocol):
    """バックグラウンド実行を抽象化する scheduler protocol。"""

    def add_task(self, func, *args, **kwargs) -> None:
        """タスクを登録する。"""


class ExperimentApplicationService:
    """実験 workflow を framework 非依存にまとめる。"""

    def __init__(self, experiment_manager, persistence_service) -> None:
        self.experiment_manager = experiment_manager
        self.persistence_service = persistence_service

    def create_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
    ) -> None:
        """実験レコードを作成する。"""
        self.persistence_service.create_experiment(
            experiment_id, experiment_name, ga_config, backtest_config
        )

    def initialize_ga_engine(self, experiment_id: str, ga_config: GAConfig) -> None:
        """GA エンジンを初期化する。"""
        if not self.experiment_manager:
            raise RuntimeError("実験管理マネージャーが初期化されていません。")
        self.experiment_manager.initialize_ga_engine(ga_config, experiment_id)

    def schedule_experiment(
        self,
        experiment_id: str,
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
        task_scheduler: TaskScheduler,
    ) -> None:
        """実験実行タスクを登録する。"""
        if not self.experiment_manager:
            raise RuntimeError("実験管理マネージャーが初期化されていません。")
        task_scheduler.add_task(
            self.experiment_manager.run_experiment,
            experiment_id,
            ga_config,
            backtest_config,
        )

    def start_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        ga_config: GAConfig,
        backtest_config: Dict[str, Any],
        task_scheduler: TaskScheduler,
    ) -> str:
        """実験作成から非同期実行登録までを行う。"""
        self.create_experiment(
            experiment_id, experiment_name, ga_config, backtest_config
        )

        try:
            self.initialize_ga_engine(experiment_id, ga_config)
            self.schedule_experiment(
                experiment_id,
                ga_config,
                backtest_config,
                task_scheduler,
            )
        except Exception as e:
            logger.warning(f"実験開始中にエラーが発生しました (experiment_id={experiment_id}): {e}")
            if self.experiment_manager:
                self.experiment_manager.release_experiment(experiment_id)
            self.persistence_service.fail_experiment(experiment_id)
            raise

        return experiment_id

    def list_experiments(self):
        """実験一覧を取得する。"""
        return self.persistence_service.list_experiments()

    def get_experiment_detail(self, experiment_id: str):
        """実験詳細を取得する。"""
        return self.persistence_service.get_experiment_detail(experiment_id)

    def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """実験停止結果を API 向け形式で返す。"""
        if not self.experiment_manager:
            return {
                "success": False,
                "message": "実験管理マネージャーが初期化されていません",
            }

        stop_result = self.experiment_manager.stop_experiment(experiment_id)
        if stop_result:
            return {
                "success": True,
                "message": "実験が正常に停止されました",
            }
        return {
            "success": False,
            "message": "実行中の実験が見つかりませんでした",
        }
