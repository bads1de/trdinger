"""
Trdinger CLI の依存サービス生成ヘルパー。
"""

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
from database.connection import SessionLocal


class SynchronousScheduler:
    """CLI 用の同期実行スケジューラ。add_task をその場で実行する。"""

    def add_task(self, func, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        func(*args, **kwargs)


def build_services() -> tuple[
    AutoStrategyService, ExperimentApplicationService
]:
    """サーバー非依存のサービス群を構築する。"""
    backtest_service = BacktestService()
    persistence_service = ExperimentPersistenceService(SessionLocal)
    experiment_manager = ExperimentManager(
        backtest_service, persistence_service
    )
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
