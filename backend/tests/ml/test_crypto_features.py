import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.feature_engineering.crypto_features import (
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
            }
        )
        df.set_index("timestamp", inplace=True)
        return df

    def test_initialization(self, calculator):
        assert isinstance(calculator, CryptoFeatureCalculator)

    def test_create_crypto_features_basic(self, calculator, sample_ohlcv_data):
        result = calculator.create_crypto_features(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        assert "price_vs_low_24h" in result.columns
        assert not result["price_vs_low_24h"].isna().all()

    def test_calculate_features_integration(self, calculator, sample_ohlcv_data):
        result = calculator.calculate_features(sample_ohlcv_data, config={})
        assert "price_vs_low_24h" in result.columns




