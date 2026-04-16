from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from app.services.auto_strategy.strategies.execution_cycle import (
    StrategyExecutionCycle,
)


class CallRecorder:
    def __init__(self, return_value=None) -> None:
        self.return_value = return_value
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value

    def assert_not_called(self) -> None:
        assert not self.calls

    def assert_called_once_with(self, *args, **kwargs) -> None:
        assert self.calls == [((args), kwargs)]

    @property
    def call_count(self) -> int:
        return len(self.calls)


@dataclass
class FakeManager:
    check_pending_order_fills: CallRecorder = field(
        default_factory=lambda: CallRecorder()
    )
    expire_pending_orders: CallRecorder = field(default_factory=lambda: CallRecorder())
    process_stateful_triggers: CallRecorder = field(
        default_factory=lambda: CallRecorder()
    )
    handle_open_position: CallRecorder = field(default_factory=lambda: CallRecorder())


@dataclass
class FakeEntryDecisionEngine:
    determine_entry_direction: CallRecorder = field(
        default_factory=lambda: CallRecorder(return_value=0.0)
    )
    execute_entry: CallRecorder = field(default_factory=lambda: CallRecorder())


@dataclass
class FakeStrategy:
    _minute_data: pd.DataFrame | None = None
    _current_bar_index: int = 3
    _is_evaluation_bar: CallRecorder = field(
        default_factory=lambda: CallRecorder(return_value=True)
    )
    _check_early_termination: CallRecorder = field(
        default_factory=lambda: CallRecorder()
    )
    position: object | None = None
    data: object | None = None
    order_manager: FakeManager = field(default_factory=FakeManager)
    stateful_conditions_evaluator: FakeManager = field(default_factory=FakeManager)
    position_manager: FakeManager = field(default_factory=FakeManager)
    entry_decision_engine: FakeEntryDecisionEngine = field(
        default_factory=FakeEntryDecisionEngine
    )


class TestStrategyExecutionCycle:
    @pytest.fixture
    def strategy(self) -> FakeStrategy:
        strategy = FakeStrategy()
        strategy.data = type(
            "Data",
            (),
            {"index": pd.date_range("2024-01-01", periods=4, freq="h")},
        )()
        return strategy

    @pytest.fixture
    def cycle(self, strategy: FakeStrategy):
        return StrategyExecutionCycle(strategy)

    def test_run_current_bar_skips_when_bar_is_before_evaluation_window(
        self, cycle, strategy: FakeStrategy
    ) -> None:
        strategy._is_evaluation_bar.return_value = False

        cycle.run_current_bar()

        strategy.order_manager.check_pending_order_fills.assert_not_called()
        strategy.order_manager.expire_pending_orders.assert_not_called()
        strategy.stateful_conditions_evaluator.process_stateful_triggers.assert_not_called()
        strategy.position_manager.handle_open_position.assert_not_called()
        strategy._check_early_termination.assert_not_called()
        strategy.entry_decision_engine.determine_entry_direction.assert_not_called()

    def test_run_current_bar_processes_pending_orders_and_triggers(
        self, cycle, strategy: FakeStrategy
    ) -> None:
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
        self, cycle, strategy: FakeStrategy
    ) -> None:
        strategy.position = object()
        strategy.position_manager.handle_open_position.return_value = True

        cycle.run_current_bar()

        strategy._check_early_termination.assert_called_once_with()
        strategy.entry_decision_engine.determine_entry_direction.assert_not_called()
        strategy.entry_decision_engine.execute_entry.assert_not_called()

    def test_run_current_bar_executes_entry_when_flat_and_signal_exists(
        self, cycle, strategy: FakeStrategy
    ) -> None:
        strategy.entry_decision_engine.determine_entry_direction.return_value = 1.0

        cycle.run_current_bar()

        strategy.entry_decision_engine.execute_entry.assert_called_once_with(1.0)
