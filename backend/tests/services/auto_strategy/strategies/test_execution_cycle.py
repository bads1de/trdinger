from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.services.auto_strategy.strategies.execution_cycle import (
    StrategyExecutionCycle,
)


class TestStrategyExecutionCycle:
    @pytest.fixture
    def strategy(self):
        strategy = MagicMock()
        strategy._minute_data = None
        strategy._current_bar_index = 3
        strategy._is_evaluation_bar.return_value = True
        strategy._check_early_termination = MagicMock()
        strategy.position = None
        strategy.data = MagicMock()
        strategy.data.index = pd.date_range("2024-01-01", periods=4, freq="h")
        strategy.order_manager = MagicMock()
        strategy.stateful_conditions_evaluator = MagicMock()
        strategy.position_exit_engine = MagicMock()
        strategy.position_exit_engine.handle_open_position.return_value = False
        strategy.entry_decision_engine = MagicMock()
        strategy.entry_decision_engine.determine_entry_direction.return_value = 0.0
        return strategy

    @pytest.fixture
    def cycle(self, strategy):
        return StrategyExecutionCycle(strategy)

    def test_run_current_bar_skips_when_bar_is_before_evaluation_window(
        self, cycle, strategy
    ):
        strategy._is_evaluation_bar.return_value = False

        cycle.run_current_bar()

        strategy.order_manager.check_pending_order_fills.assert_not_called()
        strategy.order_manager.expire_pending_orders.assert_not_called()
        strategy.stateful_conditions_evaluator.process_stateful_triggers.assert_not_called()
        strategy.position_exit_engine.handle_open_position.assert_not_called()
        strategy._check_early_termination.assert_not_called()
        strategy.entry_decision_engine.determine_entry_direction.assert_not_called()

    def test_run_current_bar_processes_pending_orders_and_triggers(
        self, cycle, strategy
    ):
        minute_data = pd.DataFrame({"Close": [100.0]})
        strategy._minute_data = minute_data

        cycle.run_current_bar()

        strategy.order_manager.check_pending_order_fills.assert_called_once_with(
            minute_data,
            strategy.data.index[-1],
            strategy._current_bar_index,
        )
        strategy.order_manager.expire_pending_orders.assert_called_once_with(
            strategy._current_bar_index
        )
        strategy.stateful_conditions_evaluator.process_stateful_triggers.assert_called_once_with()
        strategy._check_early_termination.assert_called_once_with()

    def test_run_current_bar_checks_early_termination_before_returning_after_exit(
        self, cycle, strategy
    ):
        strategy.position = MagicMock()
        strategy.position_exit_engine.handle_open_position.return_value = True

        cycle.run_current_bar()

        strategy._check_early_termination.assert_called_once_with()
        strategy.entry_decision_engine.determine_entry_direction.assert_not_called()
        strategy.entry_decision_engine.execute_entry.assert_not_called()

    def test_run_current_bar_executes_entry_when_flat_and_signal_exists(
        self, cycle, strategy
    ):
        strategy.entry_decision_engine.determine_entry_direction.return_value = 1.0

        cycle.run_current_bar()

        strategy.entry_decision_engine.execute_entry.assert_called_once_with(1.0)
