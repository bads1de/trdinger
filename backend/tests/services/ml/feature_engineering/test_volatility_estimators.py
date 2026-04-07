import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering.advanced_rolling_stats import (
    AdvancedRollingStatsCalculator,
)
from app.services.ml.feature_engineering.volatility_estimators import (
    garman_klass_volatility,
    parkinson_volatility,
    yang_zhang_volatility,
)


@pytest.fixture
def constant_ohlcv_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=30, freq="h")
    base = pd.Series(100.0, index=index)
    volume = pd.Series(1000.0, index=index)

    return pd.DataFrame(
        {
            "open": base,
            "high": base,
            "low": base,
            "close": base,
            "volume": volume,
        },
        index=index,
    )


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=60, freq="h")
    close = pd.Series(np.linspace(100.0, 130.0, len(index)), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.Series(np.maximum(open_, close) + 1.0, index=index)
    low = pd.Series(np.minimum(open_, close) - 1.0, index=index)
    volume = pd.Series(np.linspace(1000.0, 2000.0, len(index)), index=index)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_yang_zhang_volatility_constant_market_is_zero(constant_ohlcv_data):
    result = yang_zhang_volatility(
        constant_ohlcv_data["open"],
        constant_ohlcv_data["high"],
        constant_ohlcv_data["low"],
        constant_ohlcv_data["close"],
        window=5,
    )

    assert isinstance(result, pd.Series)
    assert result.iloc[:5].isna().all()
    assert np.allclose(result.dropna(), 0.0)


def test_parkinson_volatility_constant_market_is_zero(constant_ohlcv_data):
    result = parkinson_volatility(
        constant_ohlcv_data["high"],
        constant_ohlcv_data["low"],
        window=5,
    )

    assert isinstance(result, pd.Series)
    assert result.iloc[:4].isna().all()
    assert np.allclose(result.dropna(), 0.0)


def test_garman_klass_volatility_constant_market_is_zero(constant_ohlcv_data):
    result = garman_klass_volatility(
        constant_ohlcv_data["open"],
        constant_ohlcv_data["high"],
        constant_ohlcv_data["low"],
        constant_ohlcv_data["close"],
        window=5,
    )

    assert isinstance(result, pd.Series)
    assert result.iloc[:4].isna().all()
    assert np.allclose(result.dropna(), 0.0)


def test_advanced_rolling_stats_uses_shared_estimators(sample_ohlcv_data):
    calculator = AdvancedRollingStatsCalculator(windows=[5])

    result = calculator.calculate_features(sample_ohlcv_data)

    expected_columns = {
        "Yang_Zhang_Vol_5",
        "Parkinson_Vol_5",
        "Garman_Klass_Vol_5",
    }

    assert expected_columns.issubset(result.columns)
    assert np.isfinite(result[list(expected_columns)].to_numpy()).all()
