import numpy as np
import pandas as pd
import pytest
import talib

from app.services.ml.feature_engineering.technical_features import (
    TechnicalFeatureCalculator,
)


def _make_dummy_ohlcv(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(2025)
    close = 30000 + np.cumsum(rng.normal(0, 50, size=n))
    high = close + np.abs(rng.normal(20, 10, size=n))
    low = close - np.abs(rng.normal(20, 10, size=n))
    openp = close + rng.normal(0, 5, size=n)
    df = pd.DataFrame({"Open": openp, "High": high, "Low": low, "Close": close})
    return df


def test_market_regime_ma_and_extrema_align_with_talib():
    df = _make_dummy_ohlcv(200)
    calc = TechnicalFeatureCalculator()
    lookback = {"short_ma": 10, "long_ma": 50, "volatility": 20}

    # Run current implementation
    res = calc.calculate_market_regime_features(df.copy(), lookback)

    # Expected via TA-Lib
    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)

    sma_short = talib.SMA(close, timeperiod=lookback["short_ma"])
    sma_long = talib.SMA(close, timeperiod=lookback["long_ma"])

    # MAX/MIN over window (TA-Lib). Note Talib MAX/MIN default min_periods=period, unlike rolling(min_periods=1)
    # For alignment we compare last fully-formed window values and allow earlier indices to differ.
    max_20 = talib.MAX(high, timeperiod=lookback["volatility"])
    min_20 = talib.MIN(low, timeperiod=lookback["volatility"])

    # Compare last value equality where fully formed
    assert not np.isnan(sma_short[-1])
    assert not np.isnan(sma_long[-1])
    assert not np.isnan(max_20[-1])
    assert not np.isnan(min_20[-1])

    # Trend_Strength = (ma_short - ma_long)/ma_long
    expected_trend_strength_last = (sma_short[-1] - sma_long[-1]) / sma_long[-1]
    assert res["Trend_Strength"].iloc[-1] == pytest.approx(
        expected_trend_strength_last, rel=1e-9, abs=1e-8
    )

    # Range_Bound_Ratio = (Close - low20)/(high20 - low20)
    expected_range_ratio_last = (close[-1] - min_20[-1]) / (max_20[-1] - min_20[-1])
    assert res["Range_Bound_Ratio"].iloc[-1] == pytest.approx(
        expected_range_ratio_last, rel=1e-9, abs=1e-8
    )


def test_breakout_strength_uses_high_low_prev_with_talib_extrema():
    df = _make_dummy_ohlcv(200)
    calc = TechnicalFeatureCalculator()
    lookback = {"short_ma": 10, "long_ma": 50, "volatility": 20}

    res = calc.calculate_market_regime_features(df.copy(), lookback)

    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)

    max_20 = talib.MAX(high, timeperiod=lookback["volatility"])
    min_20 = talib.MIN(low, timeperiod=lookback["volatility"])

    # Expected breakout_strength last based on prev high_20/min_20
    prev_high = max_20[-2]
    prev_low = min_20[-2]
    if close[-1] > prev_high:
        expected = (close[-1] - prev_high) / prev_high
    elif close[-1] < prev_low:
        expected = (prev_low - close[-1]) / prev_low
    else:
        expected = 0.0

    assert res["Breakout_Strength"].iloc[-1] == pytest.approx(expected, rel=1e-9, abs=1e-8)

