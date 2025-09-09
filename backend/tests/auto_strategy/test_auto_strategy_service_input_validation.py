import pytest
from unittest.mock import patch, MagicMock

from fastapi import BackgroundTasks

from app.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)


@pytest.fixture
def auto_strategy_service():
    with (
        patch("app.services.auto_strategy.services.auto_strategy_service.SessionLocal"),
        patch(
            "app.services.auto_strategy.services.auto_strategy_service.BacktestDataService"
        ),
        patch(
            "app.services.auto_strategy.services.auto_strategy_service.BacktestService"
        ),
        patch(
            "app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService"
        ),
        patch(
            "app.services.auto_strategy.services.auto_strategy_service.ExperimentManager"
        ),
    ):
        service = AutoStrategyService()
        yield service


def test_start_strategy_generation_rejects_empty_experiment_id(auto_strategy_service):
    bt = BackgroundTasks()
    with pytest.raises(Exception):
        auto_strategy_service.start_strategy_generation("", "valid_name", {}, {}, bt)


def test_start_strategy_generation_rejects_none_backtest_config(auto_strategy_service):
    bt = BackgroundTasks()
    with pytest.raises(Exception):
        auto_strategy_service.start_strategy_generation("exp1", "name", {}, None, bt)


def test_start_strategy_generation_rejects_negative_values_in_ga_config(
    auto_strategy_service,
):
    bt = BackgroundTasks()
    ga = {"population_size": -5, "generations": -1}
    with pytest.raises(Exception):
        auto_strategy_service.start_strategy_generation("exp2", "name", ga, {}, bt)
