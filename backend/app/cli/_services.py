"""
Trdinger CLI の依存サービス生成ヘルパー。
"""

from __future__ import annotations

import inspect
from typing import Any

from app.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)
from app.services.auto_strategy.services.experiment_application_service import (
    ExperimentApplicationService,
    TaskScheduler,
)
from app.services.auto_strategy.services.experiment_manager import (
    ExperimentManager,
)
from app.services.auto_strategy.services.experiment_persistence_service import (
    ExperimentPersistenceService,
)
from app.services.backtest.services.backtest_service import BacktestService
from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from app.services.data_collection.orchestration.data_management_orchestration_service import (
    DataManagementOrchestrationService,
)
from database.connection import SessionLocal


class SynchronousScheduler:
    """CLI 用の同期実行スケジューラ。add_task をその場で実行する。"""

    def add_task(self, func, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        func(*args, **kwargs)


class SynchronousBackgroundTasks:
    """CLI 用の同期 BackgroundTasks 互換オブジェクト。

    FastAPI の BackgroundTasks と同じ add_task インターフェースを提供する。
    タスクは即時実行せず蓄積し、同じイベントループ内で `run_tasks()` を
    await することで順次実行する（async タスクも正しく await される）。
    """

    def __init__(self) -> None:
        self.tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def add_task(self, func: Any, *args: Any, **kwargs: Any) -> None:
        self.tasks.append((func, args, kwargs))

    async def run_tasks(self) -> None:
        for func, args, kwargs in self.tasks:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                await result


def build_data_collection_services() -> tuple[
    DataCollectionOrchestrationService, DataManagementOrchestrationService
]:
    """データ収集・管理サービスを構築する。"""
    return DataCollectionOrchestrationService(), DataManagementOrchestrationService()


def build_services() -> tuple[AutoStrategyService, ExperimentApplicationService]:
    """サーバー非依存のサービス群を構築する。"""
    backtest_service = BacktestService()
    persistence_service = ExperimentPersistenceService(SessionLocal)
    experiment_manager = ExperimentManager(backtest_service, persistence_service)
    application_service = ExperimentApplicationService(
        experiment_manager, persistence_service
    )
    auto_strategy_service = AutoStrategyService(
        backtest_service=backtest_service,
        persistence_service=persistence_service,
        experiment_manager=experiment_manager,
        experiment_application_service=application_service,
    )
    return auto_strategy_service, application_service


def build_task_scheduler() -> TaskScheduler:
    """CLI 用の同期タスクスケジューラを返す。"""
    return SynchronousScheduler()
