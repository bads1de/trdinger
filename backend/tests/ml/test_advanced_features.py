import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from backend.app.services.ml.feature_engineering.advanced_features import (
    AdvancedFeatureEngineer,
)


class TestAdvancedFeatureEngineer:
    @pytest.fixture
    def engineer(self):
        return AdvancedFeatureEngineer()

    @pytest.fixture
    def sample_ohlcv_data(self):
        dates = pd.date_range(start="2023-01-01", periods=200, freq="1H")
        df = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 200),
                "high": np.random.uniform(200, 300, 200),
                "low": np.random.uniform(50, 100, 200),
                "close": np.random.uniform(100, 200, 200),
                "volume": np.random.uniform(1000, 5000, 200),
            },
            index=dates,
        )
        return df

    def test_initialization(self, engineer):
        assert isinstance(engineer, AdvancedFeatureEngineer)

    def test_create_advanced_features_basic(self, engineer, sample_ohlcv_data):
        result = engineer.create_advanced_features(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)

        # Check for some expected columns
        assert "williams_r" in result.columns
        assert "ATR" in result.columns
        assert "Realized_Vol_20" in result.columns

    def test_add_advanced_technical_indicators(self, engineer, sample_ohlcv_data):
        result = engineer._add_advanced_technical_indicators(sample_ohlcv_data)

        # Check for indicators that will be refactored
        assert "williams_r" in result.columns
        assert "cci" in result.columns
        assert "mfi" in result.columns
        assert "ATR" in result.columns
        assert "ADX" in result.columns
        assert "Aroon_Up" in result.columns
        assert "CHOP" in result.columns

        # Check values are not all NaN (skip initial period)
        assert not result["williams_r"].iloc[50:].isna().all()

    def test_add_mtf_features(self, engineer, sample_ohlcv_data):
        result = engineer._add_mtf_features(
            sample_ohlcv_data, sample_ohlcv_data, timeframe_hours=4
        )

        # Check for MTF columns
        assert any(col.startswith("MTF_4h_") for col in result.columns)

    def test_add_range_detection_features(self, engineer, sample_ohlcv_data):
        result = engineer._add_range_detection_features(sample_ohlcv_data)
        assert "Price_Density_24h" in result.columns

    def test_add_statistical_features(self, engineer, sample_ohlcv_data):
        result = engineer._add_statistical_features(sample_ohlcv_data)
        assert "Close_std_20" in result.columns

    def test_add_volatility_features(self, engineer, sample_ohlcv_data):
        result = engineer._add_volatility_features(sample_ohlcv_data)
        assert "Realized_Vol_20" in result.columns
