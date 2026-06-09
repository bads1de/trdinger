"""
microstructure_features モジュールのユニットテスト
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering.microstructure_features import (
    MicrostructureFeatureCalculator,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=100, freq="h")
    np.random.seed(42)
    close = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5), index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": pd.Series(1000.0 + np.random.rand(100) * 500, index=index),
        },
        index=index,
    )


@pytest.fixture
def sample_fr() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {"funding_rate": np.random.randn(100) * 0.0001},
        index=index,
    )


@pytest.fixture
def sample_ls() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {"long_short_ratio": 1.0 + np.random.randn(100) * 0.1},
        index=index,
    )


class TestMicrostructureFeatureCalculator:
    def test_calculate_features_basic(self, sample_ohlcv):
        calc = MicrostructureFeatureCalculator()
        result = calc.calculate_features(sample_ohlcv)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv)
        assert result.index.equals(sample_ohlcv.index)
        assert "Amihud_Illiquidity_20h" in result.columns
        assert "Kyles_Lambda_20h" in result.columns
        assert "Returns_Kurtosis_50" in result.columns
        assert "Returns_Skewness_50" in result.columns

    def test_calculate_features_with_fr(self, sample_ohlcv, sample_fr):
        calc = MicrostructureFeatureCalculator()
        result = calc.calculate_features(sample_ohlcv, fr_df=sample_fr)

        assert "FR_Extremity_Zscore" in result.columns
        assert "FR_Change_4h" in result.columns

    def test_calculate_features_with_ls(self, sample_ohlcv, sample_ls):
        calc = MicrostructureFeatureCalculator()
        result = calc.calculate_features(sample_ohlcv, ls_df=sample_ls)

        assert "LS_Sentiment_Elasticity" in result.columns
        assert "LS_Acceleration" in result.columns
        assert "LS_Price_Incongruence" in result.columns

    def test_calculate_features_with_fr_and_ls(self, sample_ohlcv, sample_fr, sample_ls):
        calc = MicrostructureFeatureCalculator()
        result = calc.calculate_features(sample_ohlcv, fr_df=sample_fr, ls_df=sample_ls)

        assert "LS_FR_Stress_Index" in result.columns
        assert np.isfinite(result["LS_FR_Stress_Index"].iloc[50:]).all()

    def test_no_nan_in_output(self, sample_ohlcv):
        calc = MicrostructureFeatureCalculator()
        result = calc.calculate_features(sample_ohlcv)
        assert not result.isna().any().any()

    def test_amihud_illiquidity(self, sample_ohlcv):
        calc = MicrostructureFeatureCalculator()
        result = calc.calculate_amihud_illiquidity(sample_ohlcv, window=10)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        assert (result >= 0).all()

    def test_kyles_lambda(self, sample_ohlcv):
        calc = MicrostructureFeatureCalculator()
        result = calc.calculate_kyles_lambda(sample_ohlcv, window=10)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_empty_fr_df_ignored(self, sample_ohlcv):
        calc = MicrostructureFeatureCalculator()
        result = calc.calculate_features(sample_ohlcv, fr_df=pd.DataFrame())
        assert "FR_Extremity_Zscore" not in result.columns

    def test_empty_ls_df_ignored(self, sample_ohlcv):
        calc = MicrostructureFeatureCalculator()
        result = calc.calculate_features(sample_ohlcv, ls_df=pd.DataFrame())
        assert "LS_Sentiment_Elasticity" not in result.columns

    def test_constant_price_returns_zero_volatility_features(self):
        index = pd.date_range("2024-01-01", periods=50, freq="h")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            },
            index=index,
        )
        calc = MicrostructureFeatureCalculator()
        result = calc.calculate_features(df)
        assert not result.isna().any().any()
