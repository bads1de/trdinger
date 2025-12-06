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
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
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
        assert "price_vs_low_24h" in result.columns

    def test_create_price_features(self, calculator, sample_ohlcv_data):
        periods = {"short": 14}
        result = calculator._create_price_features(sample_ohlcv_data, periods)
        assert "price_vs_low_24h" in result.columns

    def test_ensure_data_quality(self, calculator, sample_ohlcv_data):
        # Introduce some NaNs
        df_with_nans = sample_ohlcv_data.copy()
        df_with_nans.loc[df_with_nans.index[0], "open_interest"] = np.nan

        result = calculator._ensure_data_quality(df_with_nans)
        assert not result["open_interest"].isna().any()
