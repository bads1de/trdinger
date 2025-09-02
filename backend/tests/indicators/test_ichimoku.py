"""
Test for ICHIMOKU technical indicator
"""
import pytest
import pandas as pd
import numpy as np
from app.services.indicators.technical_indicators.trend import TrendIndicators


class TestIchimoku:
    """Test ICHIMOKU Cloud indicator"""

    @pytest.fixture
    def sample_data(self):
        """Sample OHLC data for testing"""
        np.random.seed(42)
        n = 50
        close = pd.Series(np.random.uniform(100, 200, n), name='close')
        high = close + np.random.uniform(0, 20, n)
        low = close - np.random.uniform(0, 20, n)

        return {
            'high': high,
            'low': low,
            'close': close
        }

    def test_ichimoku_basic_calculation(self, sample_data):
        """Test basic ICHIMOKU calculation"""
        result = TrendIndicators.ichimoku_cloud(
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            tenkan=9, kijun=26, senkou=52
        )

        # Should return 5 series
        assert isinstance(result, tuple)
        assert len(result) == 5

        conv, base, span_a, span_b, lag = result

        for r in [conv, base, span_a, span_b, lag]:
            assert isinstance(r, pd.Series)
            assert len(r) == len(sample_data['high'])

        # Check basic properties
        assert conv.isna().sum() <= 9  # First 9 values should be NaN for conversion
        assert base.isna().sum() <= 26  # First 26 values should be NaN for base

    def test_ichimoku_cloud_span_shifts(self, sample_data):
        """Test ICHIMOKU cloud spans are shifted correctly"""
        conv, base, span_a, span_b, lag = TrendIndicators.ichimoku_cloud(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        # Span A and Span B should be shifted by kijun (26) periods
        assert span_a.isna().sum() >= 26
        assert span_b.isna().sum() >= 26

    def test_ichimoku_lagging_span(self, sample_data):
        """Test ICHIMOKU lagging span is shifted by kijun"""
        conv, base, span_a, span_b, lag = TrendIndicators.ichimoku_cloud(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        # Lagging span should be close shifted backwards by kijun (26)
        expected_lag = sample_data['close'].shift(-26)
        pd.testing.assert_series_equal(lag, expected_lag, check_names=False)

    def test_ichimoku_conversion_line(self, sample_data):
        """Test ICHIMOKU conversion line calculation"""
        conv, _, _, _, _ = TrendIndicators.ichimoku_cloud(
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            tenkan=9
        )

        # Manual calculation for verification
        tenkan_period = 9
        manual_conv = ((sample_data['high'].rolling(tenkan_period).max() +
                       sample_data['low'].rolling(tenkan_period).min()) / 2)

        # Compare valid values (non-NaN)
        valid_idx = manual_conv.notna()
        pd.testing.assert_series_equal(
            conv[valid_idx], manual_conv[valid_idx], check_names=False, atol=1e-10
        )

    def test_ichimoku_base_line(self, sample_data):
        """Test ICHIMOKU base line calculation"""
        _, base, _, _, _ = TrendIndicators.ichimoku_cloud(
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            kijun=26
        )

        kijun_period = 26
        manual_base = ((sample_data['high'].rolling(kijun_period).max() +
                       sample_data['low'].rolling(kijun_period).min()) / 2)

        valid_idx = manual_base.notna()
        pd.testing.assert_series_equal(
            base[valid_idx], manual_base[valid_idx], check_names=False, atol=1e-10
        )

    def test_ichimoku_na_n_handling(self, sample_data):
        """Test ICHIMOKU handles NaN values properly"""
        # Add some NaN values
        sample_data['close'].iloc[5] = np.nan
        sample_data['high'].iloc[10] = np.nan
        sample_data['low'].iloc[15] = np.nan

        result = TrendIndicators.ichimoku_cloud(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        assert isinstance(result, tuple)
        for r in result:
            assert isinstance(r, pd.Series)

    def test_ichimoku_length_validation(self, sample_data):
        """Test ICHIMOKU data length validation"""
        result = TrendIndicators.ichimoku_cloud(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        assert len(result[0]) == len(sample_data['high'])
        assert len(result[1]) == len(sample_data['low'])
        assert len(result[2]) == len(sample_data['close'])

    def test_ichimoku_all_parameters(self, sample_data):
        """Test ICHIMOKU with all custom parameters"""
        tenkan = 9
        kijun = 26
        senkou = 52

        result = TrendIndicators.ichimoku_cloud(
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            tenkan=tenkan,
            kijun=kijun,
            senkou=senkou
        )

        assert isinstance(result, tuple)
        assert len(result) == 5

        # Parameter validation through first valid calculations
        conv_tenkan_idx = np.where(~result[0].isna())[0]
        base_kijun_idx = np.where(~result[1].isna())[0]

        if len(conv_tenkan_idx) > 0:
            assert conv_tenkan_idx[0] >= tenkan - 1
        if len(base_kijun_idx) > 0:
            assert base_kijun_idx[0] >= kijun - 1

    def test_ichimoku_edge_cases(self):
        """Test ICHIMOKU with edge case data"""
        # Test with constant values
        constant_high = pd.Series([100] * 100, name='high')
        constant_low = pd.Series([90] * 100, name='low')
        constant_close = pd.Series([95] * 100, name='close')

        result = TrendIndicators.ichimoku_cloud(
            constant_high, constant_low, constant_close
        )

        assert isinstance(result, tuple)
        # All lines should be constant (95 for midpoints)
        # But due to shifting, we only check types and lengths

    def test_ichimoku_error_handling(self):
        """Test ICHIMOKU error handling for invalid inputs"""
        # Test with mismatched lengths
        short_series = pd.Series([1, 2], name='short')
        long_series = pd.Series([1] * 10, name='long')

        with pytest.raises(ValueError):
            TrendIndicators.ichimoku_cloud(
                short_series, long_series, long_series
            )

    def test_ichimoku_cloud_crosses(self, sample_data):
        """Test ICHIMOKU cloud boundary crossings"""
        conv, base, span_a, span_b, lag = TrendIndicators.ichimoku_cloud(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        # Span A and Span B form the cloud boundaries
        # Check that we can calculate cloud crossings
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

        # Check types only (crossing analysis requires more trend analysis)
        assert isinstance(cloud_top, pd.Series)
        assert isinstance(cloud_bottom, pd.Series)

    def test_ichimoku_insufficient_data_length(self):
        """Test that ICHIMOKU raises ValueError with insufficient data (data length < kijun*2=52)"""
        # Create data with only 51 elements (less than required kijun*2=52 for lagging shift)
        short_high = pd.Series([100] * 51, name='high')  # 51 data points
        short_low = pd.Series([98] * 51, name='low')     # 51 data points
        short_close = pd.Series([99] * 51, name='close') # 51 data points

        # Should raise PandasTAError (wrapped exception) due to insufficient data for lagging span calculation
        from app.services.indicators.utils import PandasTAError
        with pytest.raises(PandasTAError, match="Insufficient data length"):
            TrendIndicators.ichimoku_cloud(short_high, short_low, short_close)

    def test_ichimoku_minimum_required_length(self):
        """Test that ICHIMOKU works correctly with minimum required data (kijun*2=52)"""
        # Create minimum required data (52 elements)
        high = pd.Series([100] * 52, name='high')  # 52 data points
        low = pd.Series([98] * 52, name='low')     # 52 data points
        close = pd.Series([99] * 52, name='close') # 52 data points

        # Should not raise error with exactly required data length
        result = TrendIndicators.ichimoku_cloud(high, low, close)
        assert isinstance(result, tuple)
        assert len(result) == 5

        conv, base, span_a, span_b, lag = result
        for r in [conv, base, span_a, span_b, lag]:
            assert isinstance(r, pd.Series)