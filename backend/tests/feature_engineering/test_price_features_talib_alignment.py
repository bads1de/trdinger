import numpy as np
import pandas as pd
import pytest
import talib

from app.services.ml.feature_engineering.price_features import PriceFeatureCalculator


def _make_dummy_ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(777)
    close = 30000 + np.cumsum(rng.normal(0, 50, size=n))
    high = close + np.abs(rng.normal(20, 10, size=n))
    low = close - np.abs(rng.normal(20, 10, size=n))
    openp = close + rng.normal(0, 5, size=n)
    vol = np.abs(rng.normal(1000, 100, size=n))
    return pd.DataFrame({"Open": openp, "High": high, "Low": low, "Close": close, "Volume": vol})


def test_returns_and_stddev_and_atr_align_with_talib():
    df = _make_dummy_ohlcv()
    calc = PriceFeatureCalculator()
    lookback = {"volatility": 20}

    # run current implementation for volatility features
    res = calc.calculate_volatility_features(df.copy(), lookback)

    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)

    # TA-Lib ROCP (rate of change percent, fractional)
    rocp = talib.ROCP(close)
    # last finite value
    expected_returns_last = float(rocp[~np.isnan(rocp)][-1])
    assert res["Returns"].iloc[-1] == pytest.approx(expected_returns_last, rel=1e-9, abs=1e-8)

    # Realized volatility via STDDEV of returns over window
    stddev = talib.STDDEV(rocp, timeperiod=lookback["volatility"], nbdev=1)
    # Our implementation multiplies by sqrt(24)
    expected_realized_vol_last = float(stddev[~np.isnan(stddev)][-1]) * np.sqrt(24)
    assert res["Realized_Volatility_20"].iloc[-1] == pytest.approx(
        expected_realized_vol_last, rel=1e-9, abs=1e-8
    )

    # ATR via TA-Lib
    atr = talib.ATR(high, low, close, timeperiod=lookback["volatility"])
    expected_atr_last = float(atr[~np.isnan(atr)][-1])
    assert res["ATR_20"].iloc[-1] == pytest.approx(expected_atr_last, rel=1e-9, abs=1e-8)

