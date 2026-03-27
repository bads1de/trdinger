from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.strategies.position_exit_engine import PositionExitEngine
from app.services.auto_strategy.strategies.runtime_state import StrategyRuntimeState


class TestPositionExitEngine:
    @pytest.fixture
    def strategy(self):
        strategy = MagicMock()
        strategy.runtime_state = StrategyRuntimeState()
        strategy.position_manager = MagicMock()
        return strategy

    @pytest.fixture
    def engine(self, strategy):
        return PositionExitEngine(strategy)

    def test_handle_open_position_returns_false_without_position(self, engine, strategy):
        strategy.position = None
        strategy.runtime_state.sl_price = 95.0

        handled = engine.handle_open_position()

        assert handled is False
        strategy.position_manager.check_pessimistic_exit.assert_not_called()

    def test_handle_open_position_returns_false_without_sl(self, engine, strategy):
        strategy.position = MagicMock()
        strategy.runtime_state.sl_price = None

        handled = engine.handle_open_position()

        assert handled is False
        strategy.position_manager.check_pessimistic_exit.assert_not_called()

    def test_handle_open_position_delegates_to_position_manager(self, engine, strategy):
        strategy.position = MagicMock()
        strategy.runtime_state.sl_price = 95.0
        strategy.position_manager.check_pessimistic_exit.return_value = True

        handled = engine.handle_open_position()

        assert handled is True
        strategy.position_manager.check_pessimistic_exit.assert_called_once_with()
