import pytest
import pandas as pd
import numpy as np
import time
from app.services.ml.feature_engineering.volume_profile_features import (
    VolumeProfileFeatureCalculator,
)


class TestVolumeProfileFeatures:
    @pytest.fixture
    def sample_data(self):
        # Create a larger sample dataset to test performance
        n_samples = 2000  # Enough to trigger loops but fast enough for testing
        dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="1h")

        # Generate random OHLCV data
        np.random.seed(42)
        close = 10000 + np.cumsum(np.random.randn(n_samples) * 10)
        high = close + np.abs(np.random.randn(n_samples) * 5)
        low = close - np.abs(np.random.randn(n_samples) * 5)
        open_ = close + np.random.randn(n_samples) * 5
        volume = np.abs(np.random.randn(n_samples) * 100) + 10

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ).set_index("timestamp")

        return df

    def test_calculate_features_smoke(self, sample_data):
        """Basic smoke test to ensure features are calculated without error"""
        calculator = VolumeProfileFeatureCalculator(lookback_period=20, num_bins=10)
        features = calculator.calculate_features(sample_data, lookback_periods=[20])

        assert not features.empty
        assert "POC_Distance_20" in features.columns
        assert "HVN_Distance" in features.columns
        assert len(features) == len(sample_data)

    def test_values_correctness_basic(self):
        """Test calculation correctness on a small controllable dataset"""
        # Create very simple data: stable price, then jump
        # 10 bars at price 100, volume 100
        dates = pd.date_range(start="2023-01-01", periods=10, freq="1h")
        df = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [105] * 10,
                "low": [95] * 10,
                "close": [100] * 10,
                "volume": [100] * 10,
            },
            index=dates,
        )

        # Ensure float types (assuming Numba wants float arrays mostly)
        df = df.astype(float)

        calculator = VolumeProfileFeatureCalculator(lookback_period=5, num_bins=10)

        # Calculate features (this uses the numba optimized path)
        features = calculator.calculate_features(df, lookback_periods=[5])

        # Check last row values
        # Range 95-105. bins=10. Each bin size 1.
        # POC should be around middle (100) or uniformly distributed
        last_row = features.iloc[-1]

        # Just assert they are not NaN and within reasonable bounds
        assert not np.isnan(last_row["POC_Distance_5"])
        assert not np.isnan(last_row["VAH_Distance_5"])
        assert not np.isnan(last_row["VAL_Distance_5"])

        # In Value Area should be 1.0 because current price 100 is inside 95-105 range
        assert last_row["In_Value_Area_5"] == 1.0

    def test_performance(self, sample_data):
        """Performance test - just to print time"""
        calculator = VolumeProfileFeatureCalculator(lookback_period=50, num_bins=20)

        # Warmup (trigger JIT compilation)
        print("Warming up JIT...")
        try:
            calculator.calculate_features(sample_data.iloc[:100], lookback_periods=[50])
        except Exception as e:
            pytest.fail(f"Warmup failed: {e}")

        start_time = time.time()
        # Calculate for just one period to isolate core loop
        calculator.calculate_features(sample_data, lookback_periods=[50])
        duration = time.time() - start_time

        print(
            f"\nProcessing {len(sample_data)} rows took {duration:.4f} seconds (after warmup)"
        )
        # Assert is strict now. 2000 rows should take milliseconds with Numba.
        # Allowing 1.0s to be safe on CI envs.
        assert duration < 1.0
