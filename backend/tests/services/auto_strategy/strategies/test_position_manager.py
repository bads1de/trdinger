from types import SimpleNamespace

import pytest

from app.services.auto_strategy.strategies.position_manager import PositionManager
from app.services.auto_strategy.strategies.runtime_state import StrategyRuntimeState


class DummyPosition:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class DummyStrategy:
    def __init__(self, high: float, low: float) -> None:
        self.runtime_state = StrategyRuntimeState()
        self.position = DummyPosition()
        self.data = SimpleNamespace(Close=[100.0], High=[high], Low=[low])

    def _get_effective_tpsl_gene(self, direction: float):
        return None


class TestPositionManager:
    @pytest.mark.parametrize(
        ("direction", "high", "low"),
        [
            (1.0, 111.0, 99.0),
            (-1.0, 101.0, 89.0),
        ],
    )
    def test_handle_open_position_delegates_to_pessimistic_exit(
        self, direction: float, high: float, low: float
    ) -> None:
        strategy = DummyStrategy(high=high, low=low)
        strategy.runtime_state.set_open_position(
            entry_price=100.0,
            sl_price=None,
            tp_price=110.0,
            direction=direction,
        )
        manager = PositionManager(strategy)

        handled = manager.handle_open_position()

        assert handled is True
        assert strategy.position.close_calls == 1
        assert strategy.runtime_state.position_direction == 0.0

    def test_handle_open_position_returns_false_without_position(
        self,
    ) -> None:
        strategy = DummyStrategy(high=111.0, low=99.0)
        strategy.position = None
        strategy.runtime_state.set_open_position(
            entry_price=100.0,
            sl_price=95.0,
            tp_price=110.0,
            direction=1.0,
        )
        manager = PositionManager(strategy)

        handled = manager.handle_open_position()

        assert handled is False
        assert strategy.position is None

    def test_handle_open_position_returns_false_without_exit_levels(
        self,
    ) -> None:
        strategy = DummyStrategy(high=111.0, low=99.0)
        strategy.runtime_state.set_open_position(
            entry_price=100.0,
            sl_price=None,
            tp_price=None,
            direction=1.0,
        )
        manager = PositionManager(strategy)

        handled = manager.handle_open_position()

        assert handled is False
        assert strategy.position.close_calls == 0

    def test_check_pessimistic_exit_closes_tp_only_position(self) -> None:
        strategy = DummyStrategy(high=111.0, low=99.0)
        strategy.runtime_state.set_open_position(
            entry_price=100.0,
            sl_price=None,
            tp_price=110.0,
            direction=1.0,
        )
        manager = PositionManager(strategy)

        handled = manager.check_pessimistic_exit()

        assert handled is True
        assert strategy.position.close_calls == 1
        assert strategy.runtime_state.sl_price is None
        assert strategy.runtime_state.tp_price is None
        assert strategy.runtime_state.position_direction == 0.0
