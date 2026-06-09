"""
advanced_rolling_stats モジュールの包括的なユニットテスト
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering.advanced_rolling_stats import (
    AdvancedRollingStatsCalculator,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=120, freq="h")
    np.random.seed(42)
    close = pd.Series(100.0 + np.cumsum(np.random.randn(120) * 0.5), index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + np.abs(np.random.randn(120) * 0.3),
            "low": close - np.abs(np.random.randn(120) * 0.3),
            "close": close,
            "volume": pd.Series(1000.0 + np.random.rand(120) * 500, index=index),
        },
        index=index,
    )


class TestAdvancedRollingStatsCalculator:
    def test_default_windows(self):
        calc = AdvancedRollingStatsCalculator()
        assert calc.windows == [5, 10, 20, 50]

    def test_custom_windows(self):
        calc = AdvancedRollingStatsCalculator(windows=[3, 7])
        assert calc.windows == [3, 7]

    def test_calculate_features_basic(self, sample_ohlcv):
        calc = AdvancedRollingStatsCalculator(windows=[5, 10])
        result = calc.calculate_features(sample_ohlcv)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv)
        assert result.index.equals(sample_ohlcv.index)
        assert not result.isna().any().any()

    def test_returns_columns(self, sample_ohlcv):
        calc = AdvancedRollingStatsCalculator(windows=[5])
        result = calc.calculate_features(sample_ohlcv)

        expected = [
            "Returns_Skewness_5",
            "Returns_Kurtosis_5",
            "LogReturns_Skewness_5",
            "Volume_Skewness_5",
            "Volume_Kurtosis_5",
            "HL_Ratio_Mean_5",
            "HL_Ratio_Std_5",
            "Parkinson_Vol_5",
            "Garman_Klass_Vol_5",
            "Yang_Zhang_Vol_5",
            "Close_Position_Mean_5",
            "Close_Position_Std_5",
            "Abs_Returns_Mean_5",
            "Return_Asymmetry_5",
            "Extreme_Returns_Freq_5",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_price_volume_corr(self, sample_ohlcv):
        calc = AdvancedRollingStatsCalculator(windows=[10, 20])
        result = calc.calculate_features(sample_ohlcv)

        assert "Price_Volume_Corr_10" in result.columns
        assert "Price_Volume_Corr_20" in result.columns
        assert "Volume_Weighted_Returns_Skew_10" in result.columns

    def test_hurst_exponent(self, sample_ohlcv):
        calc = AdvancedRollingStatsCalculator(windows=[5])
        result = calc.calculate_features(sample_ohlcv)

        assert "Hurst_Exponent_100" in result.columns
        assert np.isfinite(result["Hurst_Exponent_100"].iloc[100:]).all()

    def test_multiple_windows_produce_multiple_columns(self, sample_ohlcv):
        calc = AdvancedRollingStatsCalculator(windows=[5, 10, 20])
        result = calc.calculate_features(sample_ohlcv)

        assert "Returns_Skewness_5" in result.columns
        assert "Returns_Skewness_10" in result.columns
        assert "Returns_Skewness_20" in result.columns

    def test_volume_weighted_skew(self, sample_ohlcv):
        calc = AdvancedRollingStatsCalculator(windows=[10])
        result = calc.calculate_features(sample_ohlcv)

        skew = result["Volume_Weighted_Returns_Skew_10"]
        assert np.isfinite(skew.iloc[10:]).all()

    def test_constant_market_zero_volatility(self):
        index = pd.date_range("2024-01-01", periods=60, freq="h")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000.0,
            },
            index=index,
        )
        calc = AdvancedRollingStatsCalculator(windows=[5])
        result = calc.calculate_features(df)
        assert not result.isna().any().any()
        assert (result["HL_Ratio_Mean_5"] == 0.0).all()
