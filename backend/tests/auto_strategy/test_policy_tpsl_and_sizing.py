import pytest

from app.services.auto_strategy.core.order_execution_policy import (
    OrderExecutionPolicy,
    ExecutionContext,
)
from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator


class _FactoryStub:
    def __init__(self):
        self.tpsl_calculator = TPSLCalculator()


def test_compute_tpsl_prices_long_and_short_ordering():
    factory = _FactoryStub()
    current_price = 100.0
    risk = {"stop_loss": 0.05, "take_profit": 0.10}

    # Long: SL < ENTRY < TP
    sl, tp = OrderExecutionPolicy.compute_tpsl_prices(
        factory, current_price, risk, gene=None, position_direction=1.0
    )
    assert sl is not None and tp is not None
    assert sl < current_price < tp
    assert pytest.approx(sl, rel=1e-9) == 95.0
    assert pytest.approx(tp, rel=1e-9) == 110.0

    # Short: TP < ENTRY < SL
    sl_s, tp_s = OrderExecutionPolicy.compute_tpsl_prices(
        factory, current_price, risk, gene=None, position_direction=-1.0
    )
    assert sl_s is not None and tp_s is not None
    assert tp_s < current_price < sl_s
    assert pytest.approx(sl_s, rel=1e-9) == 105.0
    assert pytest.approx(tp_s, rel=1e-9) == 90.0


def test_adjust_position_size_for_backtesting_rounding_and_fraction():
    # Integers rounding
    assert OrderExecutionPolicy.adjust_position_size_for_backtesting(2.7) == 3.0
    assert OrderExecutionPolicy.adjust_position_size_for_backtesting(-2.2) == -2.0

    # Fractions kept as-is (if > 0)
    assert OrderExecutionPolicy.adjust_position_size_for_backtesting(0.5) == 0.5
    assert OrderExecutionPolicy.adjust_position_size_for_backtesting(-0.25) == -0.25

    # Zero returns 0
    assert OrderExecutionPolicy.adjust_position_size_for_backtesting(0.0) == 0.0


def test_ensure_affordable_size_caps_units_and_fraction():
    # Units case: cap to affordable integer units
    ctx = ExecutionContext(current_price=100.0, current_equity=1000.0, available_cash=1000.0)
    size = 20.0  # 20 units would require 2000 cash
    capped = OrderExecutionPolicy.ensure_affordable_size(size, ctx)
    # 0.99 safety, max affordable units = int(990 // 100) = 9
    assert capped == 9.0

    # Fractional case: too large fraction gets scaled to 0.99
    frac_size = 0.999  # requires 999 cash > 990
    capped_frac = OrderExecutionPolicy.ensure_affordable_size(frac_size, ctx)
    assert capped_frac == pytest.approx(0.99, rel=1e-12)

    # Fractional within allowance stays
    ok_frac = 0.5  # requires 500 <= 990
    assert OrderExecutionPolicy.ensure_affordable_size(ok_frac, ctx) == ok_frac

