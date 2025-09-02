"""
Test for MIDPRICE technical indicator
"""
import pytest
import pandas as pd
import numpy as np
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.utils import PandasTAError


class TestMidprice:
    """Test MIDPRICE indicator"""

    @pytest.fixture
    def sample_data(self):
        """Sample high/low data for testing"""
        np.random.seed(42)
        n = 50
        high = pd.Series(np.random.uniform(100, 200, n), name='high')
        low = pd.Series(high - np.random.uniform(5, 50, n), name='low')

        return {
            'high': high,
            'low': low
        }

    def test_midprice_basic_calculation(self, sample_data):
        """Test basic MIDPRICE calculation"""
        result = TrendIndicators.midprice(
            sample_data['high'],
            sample_data['low'],
            length=14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['high'])

    def test_midprice_manual_calculation(self, sample_data):
        """Test MIDPRICE matches manual calculation"""
        length = 5
        result = TrendIndicators.midprice(
            sample_data['high'],
            sample_data['low'],
            length=length
        )

        # Manual calculation using pandas-ta style
        manual_midprice = ta.midprice(
            high=sample_data['high'],
            low=sample_data['low'],
            length=length
        )

        # Compare non-NaN values
        pd.testing.assert_series_equal(result, manual_midprice, check_names=False)

    def test_midprice_nan_handling(self, sample_data):
        """Test MIDPRICE handles NaN values properly"""
        # Add NaN values
        sample_data['high'].iloc[5] = np.nan
        sample_data['low'].iloc[10] = np.nan

        result = TrendIndicators.midprice(
            sample_data['high'],
            sample_data['low']
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['high'])

    def test_midprice_data_length_validation(self, sample_data):
        """Test MIDPRICE data length validation"""
        result = TrendIndicators.midprice(
            sample_data['high'],
            sample_data['low']
        )

        assert len(result) == len(sample_data['high'])

    def test_midprice_edge_cases(self):
        """Test MIDPRICE with edge case data"""
        # Test with constant values
        constant_high = pd.Series([120] * 20, name='high')
        constant_low = pd.Series([80] * 20, name='low')

        result = TrendIndicators.midprice(constant_high, constant_low)

        # Should be constant 100 (120+80)/2
        assert result.notna().all()
        assert (result == 100).all()

    def test_midprice_invalid_inputs(self):
        """Test MIDPRICE error handling for invalid inputs"""
        # Test with mismatched lengths
        short_series = pd.Series([1, 2], name='short')
        long_series = pd.Series([1] * 10, name='long')

        with pytest.raises(ValueError):
            TrendIndicators.midprice(long_series, short_series)

    def test_midprice_parameter_validation(self, sample_data):
        """Test MIDPRICE parameter validation"""
        # Test with length <= 0
        with pytest.raises(ValueError):
            TrendIndicators.midprice(
                sample_data['high'],
                sample_data['low'],
                length=0
            )

        # Test with valid length
        result = TrendIndicators.midprice(
            sample_data['high'],
            sample_data['low'],
            length=10
        )
        assert isinstance(result, pd.Series)

    def test_midprice_all_parameters(self, sample_data):
        """Test MIDPRICE with various parameters"""
        for length in [5, 10, 20]:
            result = TrendIndicators.midprice(
                sample_data['high'],
                sample_data['low'],
                length=length
            )

            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_data['high'])

            # Check NaN pattern
            if length > 1:
                assert result.iloc[:length-1].isna().all()

    def test_midprice_with_pandas_ta_fallback(self, sample_data):
        """Test MIDPRICE pandas-ta fallback behavior"""
        # This should use pandas-ta directly and match our implementation
        import pandas_ta as ta

        our_result = TrendIndicators.midprice(
            sample_data['high'],
            sample_data['low']
        )

        ta_result = ta.midprice(
            high=sample_data['high'],
            low=sample_data['low']
        )

        pd.testing.assert_series_equal(our_result, ta_result, check_names=False)

    def test_midprice_with_empty_data(self):
        """Test MIDPRICE with empty data"""
        empty_high = pd.Series([], dtype=float, name='high')
        empty_low = pd.Series([], dtype=float, name='low')

        # Should raise PandasTAError (wrapped ValueError) for empty data
        with pytest.raises(PandasTAError):
            TrendIndicators.midprice(empty_high, empty_low)

    def test_midprice_with_all_nan_data(self):
        """Test MIDPRICE with all NaN data"""
        nan_high = pd.Series([np.nan] * 20, name='high')
        nan_low = pd.Series([np.nan] * 20, name='low')

        result = TrendIndicators.midprice(nan_high, nan_low)

        # Check that it returns a series with all NaN (current behavior)
        assert isinstance(result, pd.Series)
        assert result.isna().all()  # Should return all NaN instead of failing

    def test_midprice_with_insufficient_data_length(self):
        """Test MIDPRICE with insufficient data for calculation"""
        # Data shorter than length
        short_high = pd.Series([100, 101], name='high')
        short_low = pd.Series([99, 98], name='low')

        result = TrendIndicators.midprice(short_high, short_low, length=10)

        # Should return series with all NaN values
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_midprice_with_mismatched_indices(self):
        """Test MIDPRICE with high/low having different indices"""
        high = pd.Series([100, 101, 102], index=[0, 1, 2], name='high')
        low = pd.Series([99, 98, 97], index=[1, 2, 3], name='low')  # Different index

        # This should currently fail but we'll fix it
        with pytest.raises(Exception):
            TrendIndicators.midprice(high, low)