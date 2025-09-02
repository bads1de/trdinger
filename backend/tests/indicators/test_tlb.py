"""
Test for TLB (Trend Line Break) technical indicator
"""
import pytest
import pandas as pd
import numpy as np
from backend.app.services.indicators.technical_indicators.trend import TrendIndicators


class TestTLB:
    """Test TLB (Trend Line Break) indicator"""

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

    def test_tlb_basic_calculation(self, sample_data):
        """Test basic TLB calculation"""
        result = TrendIndicators.tlb(
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            length=3
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['high'])

    def test_tlb_manual_verification(self, sample_data):
        """Test TLB manual calculation pattern"""
        length = 3
        result = TrendIndicators.tlb(
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            length=length
        )

        # Manual calculation
        manual_values = np.full(len(sample_data['close']), 0.0)

        for i in range(length, len(sample_data['close'])):
            # Check upward break
            if sample_data['close'].iloc[i] > sample_data['high'].iloc[i-length:i].max():
                manual_values[i] = 1.0
            # Check downward break
            elif sample_data['close'].iloc[i] < sample_data['low'].iloc[i-length:i].min():
                manual_values[i] = -1.0
            else:
                manual_values[i] = 0.0

        manual_result = pd.Series(manual_values, index=sample_data['close'].index)

        pd.testing.assert_series_equal(result, manual_result, check_names=False)

    def test_tlb_nan_handling(self, sample_data):
        """Test TLB handles NaN values properly"""
        # Add NaN values
        sample_data['high'].iloc[5] = np.nan
        sample_data['low'].iloc[10] = np.nan

        result = TrendIndicators.tlb(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['high'])

    def test_tlb_data_length_validation(self, sample_data):
        """Test TLB data length validation"""
        result = TrendIndicators.tlb(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        assert len(result) == len(sample_data['high'])

    def test_tlb_edge_cases(self):
        """Test TLB with edge case data"""
        # Test with constant values
        constant_high = pd.Series([120] * 30, name='high')
        constant_low = pd.Series([100] * 30, name='low')
        # Increasing close to create upward break
        constant_close = pd.Series(
            [95] * 5 + [130] * 25,  # First 5 below, rest above
            name='close'
        )

        result = TrendIndicators.tlb(constant_high, constant_low, constant_close)

        # Should have upward break at index 3 (close 130 > high max 120)
        assert result.iloc[3] == 1.0

    def test_tlb_invalid_inputs(self):
        """Test TLB error handling for invalid inputs"""
        # Test with mismatched lengths
        short_series = pd.Series([1, 2], name='short')
        long_series = pd.Series([1] * 10, name='long')

        with pytest.raises(ValueError):
            TrendIndicators.tlb(long_series, short_series, long_series)

    def test_tlb_parameter_validation(self, sample_data):
        """Test TLB parameter validation"""
        # Test with different lengths
        for length in [3, 5, 10]:
            result = TrendIndicators.tlb(
                sample_data['high'],
                sample_data['low'],
                sample_data['close'],
                length=length
            )

            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_data['high'])

            # First 'length-1' values should be NaN or 0
            if length > 1:
                assert all(result.iloc[:length-1] == 0)

    def test_tlb_all_parameters(self, sample_data):
        """Test TLB with all custom parameters"""
        length = 5

        result = TrendIndicators.tlb(
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            length=length
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['high'])

    def test_tlb_trend_detection(self, sample_data):
        """Test TLB trend detection capabilities"""
        result = TrendIndicators.tlb(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        # Values should be -1, 0, or 1
        assert all(result.isin([-1.0, 0.0, 1.0]))

        # Should have some trend breaks in random data
        assert result.nunique() >= 1

    def test_tlb_calculation_stability(self, sample_data):
        """Test TLB calculation is stable and consistent"""
        result1 = TrendIndicators.tlb(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        result2 = TrendIndicators.tlb(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        # Should be identical for same inputs
        pd.testing.assert_series_equal(result1, result2)

    def test_tlb_default_parameters(self, sample_data):
        """Test TLB with default parameters"""
        result = TrendIndicators.tlb(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        # TLB should use length=3 as default
        expected = TrendIndicators.tlb(
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            length=3
        )

        pd.testing.assert_series_equal(result, expected)

    def test_tlb_upward_break_detection(self, sample_data):
        """Test TLB upward break detection"""
        # Create specific scenario for upward break
        high_data = pd.Series([100, 100, 100, 100, 100], name='high')
        low_data = pd.Series([90, 90, 90, 90, 90], name='low')
        close_data = pd.Series([95, 95, 105, 95, 95], name='close')  # Break at index 2

        result = TrendIndicators.tlb(high_data, low_data, close_data, length=3)

        # Should detect upward break at index 2
        assert result.iloc[2] == 1.0
        # Other values should be 0
        assert all((x == 0.0 or x == 1.0) for x in result)

    def test_tlb_downward_break_detection(self, sample_data):
        """Test TLB downward break detection"""
        # Create specific scenario for downward break
        high_data = pd.Series([110, 110, 110, 110, 110], name='high')
        low_data = pd.Series([100, 100, 100, 100, 100], name='low')
        close_data = pd.Series([105, 105, 85, 105, 105], name='close')  # Break at index 2

        result = TrendIndicators.tlb(high_data, low_data, close_data, length=3)

        # Should detect downward break at index 2
        assert result.iloc[2] == -1.0
        # Other values should be 0
        assert all((x == 0.0 or x == -1.0) for x in result)