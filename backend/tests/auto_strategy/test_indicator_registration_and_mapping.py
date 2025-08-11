import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.calculators.indicator_calculator import IndicatorCalculator


class _FakeData:
    def __init__(self, n=120):
        idx = pd.date_range("2024-02-01", periods=n, freq="H")
        close = np.linspace(50, 70, n)
        self.df = pd.DataFrame({
            "Close": close.astype(float),
        }, index=idx)

    @property
    def Close(self):
        return self.df["Close"]


class _FakeStrategy:
    def __init__(self, data: _FakeData):
        self.data = data
    def I(self, f):
        return f()


def test_indicator_registration_names_macd_bb():
    calc = IndicatorCalculator()
    s = _FakeStrategy(_FakeData())

    # MACD: 3出力想定 -> MACD_0, MACD_1, MACD_2
    macd = calc.calculate_indicator(s.data, "MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9})
    assert isinstance(macd, tuple) and len(macd) == 3

    # IndicatorCalculator.init_indicatorの登録規約に従って属性付与
    for i, arr in enumerate(macd):
        setattr(s, f"MACD_{i}", arr)
    assert hasattr(s, "MACD_0") and hasattr(s, "MACD_1") and hasattr(s, "MACD_2")

    # BB: 3出力想定 -> BB_0, BB_1, BB_2
    bb = calc.calculate_indicator(s.data, "BB", {"period": 20, "std": 2.0})
    assert isinstance(bb, tuple) and len(bb) == 3
    for i, arr in enumerate(bb):
        setattr(s, f"BB_{i}", arr)
    assert hasattr(s, "BB_0") and hasattr(s, "BB_1") and hasattr(s, "BB_2")

