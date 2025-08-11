import numpy as np
import pandas as pd

from app.services.auto_strategy.calculators.indicator_calculator import (
    IndicatorCalculator,
)


class _FakeStrategy:
    def __init__(self, df):
        class _D:
            pass

        self.data = _D()
        self.data.Open = df["Open"].values
        self.data.High = df["High"].values
        self.data.Low = df["Low"].values
        self.data.Close = df["Close"].values
        self._vals = {}

    def I(self, f):
        # backtestingのI相当: 配列をそのまま返す
        return f()


def test_bb_macd_resolution_mapping():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    base = np.linspace(100, 110, n)
    df = pd.DataFrame(
        {
            "Open": base,
            "High": base * 1.001,
            "Low": base * 0.999,
            "Close": base,
            "Volume": 1000,
        },
        index=idx,
    )

    calc = IndicatorCalculator()
    s = _FakeStrategy(df)
    # IndicatorCalculator expects backtesting-like data with .df
    s.data.df = df

    # MACD (3 outputs)
    macd = calc.calculate_indicator(
        s.data, "MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}
    )
    s.MACD_0, s.MACD_1, s.MACD_2 = macd

    # BB (3 outputs)
    bb = calc.calculate_indicator(s.data, "BB", {"period": 20, "stddev": 2.0})
    s.BB_0, s.BB_1, s.BB_2 = bb

    # ConditionEvaluatorのget_condition_value互換: ここでは直接属性で確認
    assert hasattr(s, "MACD_0") and hasattr(s, "BB_1")
    # 想定される名称: BB_Upper_{period}/BB_Middle_{period}/BB_Lower_{period}
    # IndicatorCalculatorは (upper, middle, lower) の順で返すため、BB_0>=BB_1>=BB_2
    assert s.BB_0[-1] >= s.BB_1[-1] >= s.BB_2[-1]
