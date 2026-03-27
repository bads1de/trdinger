from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.services.experiment_application_service import (
    ExperimentApplicationService,
)


class RecordingTaskScheduler:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs) -> None:
        self.tasks.append((func, args, kwargs))


@pytest.fixture
def app_service():
    manager = MagicMock()
    persistence = MagicMock()
    return ExperimentApplicationService(manager, persistence)


def test_start_experiment_schedules_background_run(app_service):
    scheduler = RecordingTaskScheduler()
    ga_config = GAConfig.from_dict({})
    backtest_config = {"symbol": "BTC/USDT:USDT", "timeframe": "1h"}

    result = app_service.start_experiment(
        "exp-001",
        "Experiment",
        ga_config,
        backtest_config,
        scheduler,
    )

    assert result == "exp-001"
    app_service.persistence_service.create_experiment.assert_called_once_with(
        "exp-001",
        "Experiment",
        ga_config,
        backtest_config,
    )
    app_service.experiment_manager.initialize_ga_engine.assert_called_once_with(
        ga_config, "exp-001"
    )
    assert len(scheduler.tasks) == 1
    func, args, kwargs = scheduler.tasks[0]
    assert func == app_service.experiment_manager.run_experiment
    assert args == ("exp-001", ga_config, backtest_config)
    assert kwargs == {}


def test_start_experiment_cleans_up_on_schedule_error(app_service):
    class FailingTaskScheduler:
        def add_task(self, func, *args, **kwargs) -> None:
            raise RuntimeError("boom")

    ga_config = GAConfig.from_dict({})

    with pytest.raises(RuntimeError):
        app_service.start_experiment(
            "exp-001",
            "Experiment",
            ga_config,
            {"symbol": "BTC/USDT:USDT"},
            FailingTaskScheduler(),
        )

    app_service.experiment_manager.release_experiment.assert_called_once_with("exp-001")
    app_service.persistence_service.fail_experiment.assert_called_once_with("exp-001")


def test_stop_experiment_maps_manager_result(app_service):
    app_service.experiment_manager.stop_experiment.return_value = True

    result = app_service.stop_experiment("exp-001")

    assert result == {
        "success": True,
        "message": "実験が正常に停止されました",
    }
