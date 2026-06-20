"""
runtime_state モジュールのユニットテスト
"""


from app.services.auto_strategy.strategies.runtime_state import StrategyRuntimeState


class TestStrategyRuntimeState:
    def test_initial_state(self):
        state = StrategyRuntimeState()
        assert state.sl_price is None
        assert state.tp_price is None
        assert state.entry_price is None
        assert state.position_direction == 0.0
        assert state.tp_reached is False
        assert state.trailing_tp_sl is None

    def test_set_open_position(self):
        state = StrategyRuntimeState()
        state.set_open_position(
            entry_price=100.0,
            sl_price=95.0,
            tp_price=110.0,
            direction=1.0,
        )
        assert state.entry_price == 100.0
        assert state.sl_price == 95.0
        assert state.tp_price == 110.0
        assert state.position_direction == 1.0
        assert state.tp_reached is False
        assert state.trailing_tp_sl is None

    def test_set_open_position_resets_tp_reached(self):
        state = StrategyRuntimeState()
        state.tp_reached = True
        state.set_open_position(
            entry_price=100.0,
            sl_price=95.0,
            tp_price=110.0,
            direction=1.0,
        )
        assert state.tp_reached is False

    def test_set_open_position_resets_trailing(self):
        state = StrategyRuntimeState()
        state.trailing_tp_sl = 105.0
        state.set_open_position(
            entry_price=100.0,
            sl_price=95.0,
            tp_price=110.0,
            direction=-1.0,
        )
        assert state.trailing_tp_sl is None

    def test_reset_position(self):
        state = StrategyRuntimeState()
        state.set_open_position(
            entry_price=100.0,
            sl_price=95.0,
            tp_price=110.0,
            direction=1.0,
        )
        state.reset_position()

        assert state.sl_price is None
        assert state.tp_price is None
        assert state.entry_price is None
        assert state.position_direction == 0.0
        assert state.tp_reached is False
        assert state.trailing_tp_sl is None

    def test_runtime_state_marker(self):
        assert StrategyRuntimeState.__runtime_state_marker__ is True

    def test_no_position_direction_default(self):
        state = StrategyRuntimeState()
        assert state.position_direction == 0.0
