import asyncio
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import app.services.ml.orchestration.ml_training_orchestration_service as orchestration_module
from app.services.ml.orchestration.ml_training_orchestration_service import (
    training_status,
    training_status_lock,
)


BASE_TRAINING_STATUS = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "message": "待機中",
    "start_time": None,
    "end_time": None,
    "model_info": None,
    "error": None,
    "training_id": None,
    "task_id": None,
}


@pytest.fixture(autouse=True)
def reset_training_status():
    with training_status_lock:
        original = deepcopy(training_status)
        training_status.clear()
        training_status.update(BASE_TRAINING_STATUS)

    try:
        yield
    finally:
        with training_status_lock:
            training_status.clear()
            training_status.update(original)


@pytest.fixture
def service():
    with patch.object(orchestration_module, "EnsembleTrainer") as mock_trainer_cls:
        with patch.object(orchestration_module, "OptimizationService") as mock_opt_cls:
            mock_trainer_cls.return_value = MagicMock()
            mock_opt_cls.return_value = MagicMock()
            yield orchestration_module.MLTrainingService()


@pytest.fixture
def training_config():
    return SimpleNamespace(
        symbol="BTCUSDT",
        timeframe="1h",
        start_date="2024-01-01T00:00:00",
        end_date="2024-01-15T00:00:00",
        train_test_split=0.8,
        validation_split=0.2,
        prediction_horizon=12,
        cross_validation_folds=3,
        threshold_method="TRIPLE_BARRIER",
        threshold_up=0.02,
        threshold_down=0.01,
        quantile_threshold=0.5,
        random_state=42,
        early_stopping_rounds=10,
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1,
        save_model=False,
        optimization_settings=None,
        ensemble_config=SimpleNamespace(model_dump=lambda: {"enabled": True}),
        single_model_config=None,
    )


@pytest.fixture
def training_data():
    index = pd.date_range("2024-01-01", periods=20, freq="h")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 119, 20),
            "high": np.linspace(101, 120, 20),
            "low": np.linspace(99, 118, 20),
            "close": np.linspace(100.5, 119.5, 20),
            "volume": np.linspace(1000, 2000, 20),
        },
        index=index,
    )


def test_build_training_params_includes_timeframe(service, training_config):
    params = service._build_training_params(training_config)

    assert params["timeframe"] == "1h"
    assert params["horizon_n"] == 12
    assert params["use_cross_validation"] is True
    assert params["cv_splits"] == 3
    assert params["test_size"] == pytest.approx(0.2)


def test_start_training_schedules_background_task(service, training_config):
    background_tasks = MagicMock()
    db = MagicMock()

    result = asyncio.run(service.start_training(training_config, background_tasks, db))

    background_tasks.add_task.assert_called_once()
    args = background_tasks.add_task.call_args.args

    assert len(args) == 3
    assert args[0] == service._train_in_background
    assert args[1] is training_config
    assert isinstance(args[2], str)
    assert args[2].startswith("training_")
    assert result["success"] is True


def test_execute_actual_training_marks_error_when_train_model_fails(
    service, training_config, training_data
):
    with training_status_lock:
        training_status.update(
            {
                "is_training": True,
                "progress": 0,
                "status": "starting",
                "message": "開始",
                "start_time": "2024-01-01T00:00:00",
                "end_time": None,
                "model_info": None,
                "error": None,
                "training_id": "training_001",
                "task_id": "task_001",
            }
        )

    service.train_model = MagicMock(
        return_value={"success": False, "message": "boom"}
    )

    with patch.object(orchestration_module, "EnsembleTrainer") as mock_trainer_cls:
        mock_trainer_cls.return_value = MagicMock()
        service._execute_actual_training(
            "ensemble",
            {"enabled": True},
            None,
            training_config,
            training_data,
            {"timeframe": "1h"},
        )

    with training_status_lock:
        assert training_status["status"] == "error"
        assert training_status["is_training"] is False
        assert training_status["progress"] == 100
        assert training_status["message"] == "boom"
        assert training_status["error"] == "boom"
        assert training_status["model_info"] == {"success": False, "message": "boom"}


def test_train_in_background_closes_own_session(service, training_config, training_data):
    fake_session = MagicMock()
    mock_data_service = MagicMock()
    mock_data_service.get_ml_training_data.return_value = training_data

    with training_status_lock:
        training_status.update(
            {
                "is_training": True,
                "progress": 0,
                "status": "starting",
                "message": "開始",
                "start_time": "2024-01-01T00:00:00",
                "end_time": None,
                "model_info": None,
                "error": None,
                "training_id": "training_002",
                "task_id": None,
            }
        )

    with patch("database.connection.get_session", return_value=fake_session):
        with patch.object(
            orchestration_module.background_task_manager,
            "managed_task",
            return_value=nullcontext("task-1"),
        ):
            with patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.OHLCVRepository"
            ) as mock_ohlcv_repo:
                with patch(
                    "app.services.ml.orchestration.ml_training_orchestration_service.OpenInterestRepository"
                ) as mock_oi_repo:
                    with patch(
                        "app.services.ml.orchestration.ml_training_orchestration_service.FundingRateRepository"
                    ) as mock_fr_repo:
                        with patch(
                            "app.services.backtest.backtest_data_service.BacktestDataService",
                            return_value=mock_data_service,
                        ) as mock_backtest_cls:
                            with patch.object(
                                service, "_execute_actual_training"
                            ) as mock_execute:
                                asyncio.run(
                                    service._train_in_background(
                                        training_config, "training_002"
                                    )
                                )

    mock_ohlcv_repo.assert_called_once_with(fake_session)
    mock_oi_repo.assert_called_once_with(fake_session)
    mock_fr_repo.assert_called_once_with(fake_session)
    mock_backtest_cls.assert_called_once()
    mock_execute.assert_called_once()
    fake_session.close.assert_called_once()
