import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from backend.app.services.ml.feature_engineering.crypto_features import (
    CryptoFeatureCalculator,
)


class TestCryptoFeatureCalculator:
    @pytest.fixture
    def calculator(self):
        return CryptoFeatureCalculator()

    @pytest.fixture
    def sample_ohlcv_data(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(200, 300, 100),
                "low": np.random.uniform(50, 100, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000, 5000, 100),
                "open_interest": np.random.uniform(500, 1000, 100),
                "funding_rate": np.random.uniform(-0.01, 0.01, 100),
            }
        )
        df.set_index("timestamp", inplace=True)
        return df

    def test_initialization(self, calculator):
        assert isinstance(calculator, CryptoFeatureCalculator)
        assert "price" in calculator.feature_groups
        assert "volume" in calculator.feature_groups

    def test_create_crypto_features_basic(self, calculator, sample_ohlcv_data):
        result = calculator.create_crypto_features(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        # Check if some expected columns are present
        assert "price_vs_high_24h" in result.columns
        assert "volume_change_short" in result.columns

    def test_create_technical_features(self, calculator, sample_ohlcv_data):
        periods = {"short": 14, "medium": 24, "long": 72}
        result = calculator._create_technical_features(sample_ohlcv_data, periods)

        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

        # Check values are not all NaN (skip first few due to calculation window)
        assert not result["macd"].iloc[50:].isna().all()

    def test_create_price_features(self, calculator, sample_ohlcv_data):
        periods = {"short": 14}
        result = calculator._create_price_features(sample_ohlcv_data, periods)
        assert "price_vs_high_24h" in result.columns
        assert "price_vs_low_24h" in result.columns

    def test_create_volume_features(self, calculator, sample_ohlcv_data):
        periods = {"short": 14, "medium": 24}
        result = calculator._create_volume_features(sample_ohlcv_data, periods)
        assert "volume_change_short" in result.columns
        assert "price_vs_vwap_24h" in result.columns

    def test_create_open_interest_features(self, calculator, sample_ohlcv_data):
        periods = {"medium": 24}
        result = calculator._create_open_interest_features(sample_ohlcv_data, periods)
        assert "oi_change_medium" in result.columns
        assert "oi_price_divergence" in result.columns

    def test_create_funding_rate_features(self, calculator, sample_ohlcv_data):
        periods = {"short": 14}
        # Patch where the class is defined
        with patch(
            "backend.app.services.ml.feature_engineering.funding_rate_features.FundingRateFeatureCalculator"
        ) as MockFRCalc:
            mock_instance = MockFRCalc.return_value
            # Return original df with dummy FR features
            mock_df = sample_ohlcv_data.copy()
            mock_df["fr_dummy"] = 1.0
            mock_instance.calculate_features.return_value = mock_df

            result = calculator._create_funding_rate_features(
                sample_ohlcv_data, periods
            )

            assert "fr_dummy" in result.columns
            assert "fr_dummy" in calculator.feature_groups["funding_rate"]

    def test_create_composite_features(self, calculator, sample_ohlcv_data):
        periods = {}
        result = calculator._create_composite_features(sample_ohlcv_data, periods)
        assert "volume_price_efficiency" in result.columns

    def test_ensure_data_quality(self, calculator, sample_ohlcv_data):
        # Introduce some NaNs
        df_with_nans = sample_ohlcv_data.copy()
        df_with_nans.loc[df_with_nans.index[0], "open_interest"] = np.nan

        result = calculator._ensure_data_quality(df_with_nans)
        assert not result["open_interest"].isna().any()
