import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.calculators.position_sizing_helper import (
    PositionSizingHelper,
)
from app.services.indicators.technical_indicators.volatility import (
    VolatilityIndicators,
)


def _make_dummy_ohlcv(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    close = 30000 + np.cumsum(rng.normal(0, 50, size=n))
    high = close + np.abs(rng.normal(20, 10, size=n))
    low = close - np.abs(rng.normal(20, 10, size=n))
    openp = close + rng.normal(0, 5, size=n)
    df = pd.DataFrame({"Open": openp, "High": high, "Low": low, "Close": close})
    return df


def test_calculate_atr_matches_talib_last_value():
    # Arrange
    df = _make_dummy_ohlcv(200)
    helper = PositionSizingHelper()
    period = 14

    # Act
    atr_value = helper._calculate_atr_from_data(df, period=period)

    # Expected via TA-Lib-backed VolatilityIndicators
    atr_arr = VolatilityIndicators.atr(
        df["High"].values, df["Low"].values, df["Close"].values, period=period
    )
    finite = atr_arr[~np.isnan(atr_arr)]
    assert finite.size > 0, "ATR array should contain at least one finite value"
    expected_last = float(finite[-1])

    # Assert (tight tolerance; should match TA-Lib's last value)
    assert atr_value == pytest.approx(expected_last, rel=1e-9, abs=1e-8)


def test_prepare_market_data_uses_calculated_atr_when_available():
    # Arrange
    df = _make_dummy_ohlcv(120)
    helper = PositionSizingHelper()
    current_price = float(df["Close"].iloc[-1])

    # Act
    market_data = helper.prepare_market_data_for_position_sizing(df, current_price)

    # Assert
    assert market_data.get("atr_source") == "calculated"
    assert market_data["atr"] > 0
    assert market_data["atr_pct"] == pytest.approx(
        market_data["atr"] / current_price, rel=1e-9
    )


def test_prepare_market_data_fallback_when_insufficient_length():
    # Arrange: shorter than typical ATR period -> should fallback to estimate
    df = _make_dummy_ohlcv(10)
    helper = PositionSizingHelper()
    current_price = 30000.0

    # Act
    market_data = helper.prepare_market_data_for_position_sizing(df, current_price)

    # Assert
    assert market_data.get("atr_source") == "estimated"
    assert market_data["atr_pct"] == 0.04
    assert market_data["atr"] == pytest.approx(current_price * 0.04, rel=1e-9)

