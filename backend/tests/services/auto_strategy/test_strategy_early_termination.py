from types import SimpleNamespace

import pandas as pd
import pytest

from app.services.auto_strategy.strategies.early_termination import (
    StrategyEarlyTermination,
    StrategyEarlyTerminationController,
)


class TestStrategyEarlyTerminationController:
    def _build_strategy(self) -> SimpleNamespace:
        index = pd.date_range("2024-01-01 00:00:00", periods=10, freq="h")
        data = SimpleNamespace(index=index)
        return SimpleNamespace(
            data=data,
            _evaluation_start=None,
            _evaluation_index=None,
            _evaluation_start_index=0,
            _evaluation_total_bars=10,
            _total_bars=10,
            _current_bar_index=0,
            _starting_equity=10000.0,
            _max_equity_seen=10000.0,
            equity=10000.0,
            closed_trades=[],
            enable_early_termination=False,
            early_termination_max_drawdown=None,
            early_termination_min_trades=None,
            early_termination_min_trade_check_progress=0.5,
            early_termination_trade_pace_tolerance=0.5,
            early_termination_min_expectancy=None,
            early_termination_expectancy_min_trades=5,
            early_termination_expectancy_progress=0.6,
        )

    def test_get_progress_ratio_uses_evaluation_window(self):
        strategy = self._build_strategy()
        controller = StrategyEarlyTerminationController(strategy)
        strategy._evaluation_start = pd.Timestamp("2024-01-01 08:00:00")
        strategy._evaluation_index = pd.DatetimeIndex(strategy.data.index)
        strategy._evaluation_start_index = 8
        strategy._evaluation_total_bars = 2
        strategy.data.index = strategy.data.index[:9]
        strategy._current_bar_index = 9

        assert controller.get_progress_ratio() == pytest.approx(0.5)

    def test_is_evaluation_bar_aligns_timezone_mismatch(self):
        strategy = self._build_strategy()
        controller = StrategyEarlyTerminationController(strategy)
        strategy.data.index = pd.date_range(
            "2024-01-01 00:00:00",
            periods=1,
            freq="h",
            tz="UTC",
        )
        strategy._evaluation_start = pd.Timestamp("2024-01-01 01:00:00")

        assert controller.is_evaluation_bar() is False

    def test_should_terminate_early_on_drawdown(self):
        strategy = self._build_strategy()
        controller = StrategyEarlyTerminationController(strategy)
        strategy.enable_early_termination = True
        strategy.early_termination_max_drawdown = 0.1
        strategy.equity = 8800.0
        strategy._current_bar_index = 5

        assert controller.should_terminate_early() == "max_drawdown"

    def test_should_terminate_early_on_expectancy(self):
        strategy = self._build_strategy()
        controller = StrategyEarlyTerminationController(strategy)
        strategy.enable_early_termination = True
        strategy.early_termination_min_expectancy = -0.01
        strategy.early_termination_expectancy_min_trades = 2
        strategy.early_termination_expectancy_progress = 0.6
        strategy._current_bar_index = 8
        strategy.closed_trades = [
            SimpleNamespace(pl_pct=-0.03),
            SimpleNamespace(pl_pct=0.0),
        ]

        assert controller.should_terminate_early() == "expectancy"

    def test_check_early_termination_raises_strategy_exception(self):
        strategy = self._build_strategy()
        controller = StrategyEarlyTerminationController(strategy)
        strategy.enable_early_termination = True
        strategy.early_termination_max_drawdown = 0.1
        strategy.equity = 8800.0

        with pytest.raises(StrategyEarlyTermination, match="max_drawdown"):
            controller.check_early_termination()
