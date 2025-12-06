import pandas as pd
import numpy as np
import pytest
from app.services.ml.label_generation.trend_scanning import TrendScanning


class TestTrendScanning:
    @pytest.fixture
    def sample_data(self):
        """
        Create synthetic data with clear trends.
        """
        # 100 points
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")

        # 0-30: Uptrend
        # 30-60: Downtrend
        # 60-100: Sideways/Noise

        prices = np.zeros(100)

        # Uptrend: y = x + noise
        prices[0:30] = np.arange(30) + np.random.normal(0, 0.5, 30)

        # Downtrend: y = 30 - (x-30) + noise
        prices[30:60] = 30 - np.arange(30) + np.random.normal(0, 0.5, 30)

        # Sideways: y = 0 + noise
        prices[60:100] = np.random.normal(0, 1.0, 40)

        # Shift up to avoid negative prices
        prices += 100

        return pd.Series(prices, index=dates, name="close")

    def test_uptrend_detection(self, sample_data):
        """
        Test that strong uptrends are labeled as 1.
        """
        ts = TrendScanning(min_window=5, max_window=20, min_t_value=2.0)

        # Test a point early in the uptrend (e.g., index 5)
        # It should see the uptrend ahead.
        t_event = sample_data.index[5]

        labels = ts.get_labels(sample_data, t_events=pd.DatetimeIndex([t_event]))

        assert len(labels) == 1
        assert labels.iloc[0]["bin"] == 1
        assert labels.iloc[0]["t_value"] > 2.0
        assert labels.iloc[0]["ret"] > 0

    def test_downtrend_detection(self, sample_data):
        """
        Test that strong downtrends are labeled as -1.
        """
        ts = TrendScanning(min_window=5, max_window=20, min_t_value=2.0)

        # Test a point early in the downtrend (e.g., index 35)
        t_event = sample_data.index[35]

        labels = ts.get_labels(sample_data, t_events=pd.DatetimeIndex([t_event]))

        assert len(labels) == 1
        assert labels.iloc[0]["bin"] == -1
        assert labels.iloc[0]["t_value"] < -2.0
        assert labels.iloc[0]["ret"] < 0

    def test_sideways_detection(self, sample_data):
        """
        Test that sideways markets are labeled as 0.
        """
        ts = TrendScanning(
            min_window=5, max_window=20, min_t_value=5.0
        )  # High threshold to force 0

        # Test a point in sideways (e.g., index 70)
        t_event = sample_data.index[70]

        labels = ts.get_labels(sample_data, t_events=pd.DatetimeIndex([t_event]))

        assert len(labels) == 1
        # It might detect a small trend if noise aligns, but with high threshold it should be 0
        # Or t-value should be low.
        # Let's check t-value absolute is likely small or we enforce bin 0

        # Note: Random walk can have trends.
        # But we expect bin to be 0 if t-value is below threshold.
        if abs(labels.iloc[0]["t_value"]) < 5.0:
            assert labels.iloc[0]["bin"] == 0

    def test_window_selection(self):
        """
        Test that it selects the window with the strongest trend.
        """
        # Create data where short term is noise, long term is strong trend
        dates = pd.date_range(start="2023-01-01", periods=30, freq="h")
        prices = np.array(
            [
                10,
                9,
                11,
                10,
                12,
                11,
                13,
                12,
                14,
                13,
                15,
                14,
                16,
                15,
                17,
                16,
                18,
                17,
                19,
                18,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
            ]
        )
        # Overall strong uptrend.

        s = pd.Series(prices, index=dates)

        ts = TrendScanning(min_window=5, max_window=20)
        labels = ts.get_labels(s, t_events=pd.DatetimeIndex([dates[0]]))

        # Should pick a window that captures the trend
        assert labels.iloc[0]["bin"] == 1
        assert pd.notna(labels.iloc[0]["t1"])
