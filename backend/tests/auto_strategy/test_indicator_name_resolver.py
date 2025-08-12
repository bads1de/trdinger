import numpy as np

from app.services.auto_strategy.core.indicator_name_resolver import IndicatorNameResolver


class _Dummy:
    def __init__(self):
        self.data = type("D", (), {
            "Close": np.array([1, 2, 3, 4, 5], dtype=float),
            "Open": np.array([1, 1, 1, 1, 1], dtype=float),
            "High": np.array([1, 2, 3, 4, 6], dtype=float),
            "Low": np.array([1, 1, 2, 3, 4], dtype=float),
        })()
        # 一部の指標属性
        self.MACD_0 = np.array([np.nan, 0, 1, 2, 3], dtype=float)
        self.BB_1 = np.array([10, 11, 12, 13, 14], dtype=float)
        self.SMA = np.array([1, 1, 2, 3, 5], dtype=float)
        self.KELTNER_1 = np.array([20, 21, 22, 23, 24], dtype=float)
        self.STOCH_0 = np.array([30, 40, 50, 60, 70], dtype=float)


def test_resolve_price_and_numeric():
    s = _Dummy()
    ok, v = IndicatorNameResolver.try_resolve_value("close", s)
    assert ok and v == 5
    ok, v = IndicatorNameResolver.try_resolve_value("1.23", s)
    assert ok and abs(v - 1.23) < 1e-9


def test_resolve_macd_default():
    s = _Dummy()
    ok, v = IndicatorNameResolver.try_resolve_value("MACD", s)
    assert ok and v == 3


def test_resolve_bb_middle_alias():
    s = _Dummy()
    ok, v = IndicatorNameResolver.try_resolve_value("BB_Middle_20", s)
    assert ok and v == 14


def test_resolve_keltner_default():
    s = _Dummy()
    ok, v = IndicatorNameResolver.try_resolve_value("KELTNER", s)
    assert ok and v == 24


def test_resolve_sma_unified():
    s = _Dummy()
    ok, v = IndicatorNameResolver.try_resolve_value("SMA_14", s)
    assert ok and v == 5

