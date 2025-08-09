import numpy as np
import pandas as pd
import pytest

import pandas_ta as ta

from app.services.indicators.technical_indicators.trend import TrendIndicators


def make_price_series(n=200):
    base = np.linspace(100.0, 120.0, n)
    high = base + 1.0
    low = base - 1.0
    close = base + 0.2
    return high.astype(float), low.astype(float), close.astype(float)


def test_sarext_returns_array():
    high, low, _ = make_price_series(100)
    res = TrendIndicators.sarext(high, low)
    assert hasattr(res, "__len__")
    assert len(res) == len(high)


def test_midpoint_and_midprice_return_lengths():
    high, low, close = make_price_series(120)
    midp = TrendIndicators.midpoint(close, period=10)
    assert len(midp) == len(close)

    mp = TrendIndicators.midprice(high, low, period=10)
    assert len(mp) == len(high)


def test_ht_trendline_if_available():
    # pandas-ta may or may not expose ht_trendline depending on version.
    if not hasattr(ta, "ht_trendline"):
        pytest.skip("pandas-ta does not expose ht_trendline in this environment")

    _, _, close = make_price_series(150)
    res = TrendIndicators.ht_trendline(close)
    assert len(res) == len(close)


def test_mavp_raises_not_implemented():
    # MAVP is not implemented via pandas-ta wrapper; should raise NotImplementedError
    data = np.linspace(1.0, 2.0, 50).astype(float)
    periods = np.full(50, 5.0)
    with pytest.raises(NotImplementedError):
        TrendIndicators.mavp(data, periods)
