"""
Test for T3 technical indicator
"""
import pytest
import pandas as pd
import numpy as np
import pandas_ta as ta
from backend.app.services.indicators.technical_indicators.trend import TrendIndicators


class TestT3:
    """Test T3 (Tillson T3 Moving Average) indicator"""

    @pytest.fixture
    def sample_data(self):
        """Sample close data for testing"""
        np.random.seed(42)
        n = 50
        close = pd.Series(np.random.uniform(100, 200, n), name='close')

        return close

    def test_t3_basic_calculation(self, sample_data):
        """Test basic T3 calculation"""
        result = TrendIndicators.t3(
            sample_data,
            length=5,
            a=0.7
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_t3_manual_verification(self, sample_data):
        """Test T3 matches known calculation pattern"""
        length = 5
        a = 0.7

        result = TrendIndicators.t3(
            sample_data,
            length=length,
            a=a
        )

        # T3 uses pandas-ta implementation, verify it returns valid values
        ta_result = ta.t3(
            sample_data,
            length=length,
            a=a
        )

        # Compare non-NaN values
        mask = result.notna() & ta_result.notna()
        pd.testing.assert_series_equal(
            result[mask],
            ta_result[mask],
            check_names=False,
            atol=1e-10
        )

    def test_t3_nan_handling(self, sample_data):
        """Test T3 handles NaN values properly"""
        # Add NaN values
        sample_data.iloc[5] = np.nan

        result = TrendIndicators.t3(sample_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_t3_data_length_validation(self, sample_data):
        """Test T3 data length validation"""
        result = TrendIndicators.t3(sample_data)

        assert len(result) == len(sample_data)

    def test_t3_edge_cases(self):
        """Test T3 with edge case data"""
        # Test with constant values
        constant_data = pd.Series([100] * 30, name='close')

        result = TrendIndicators.t3(constant_data)

        # Should be constant 100
        assert result.notna().all()
        assert (result == 100).all()

    def test_t3_invalid_inputs(self):
        """Test T3 error handling for invalid inputs"""
        # Test with None data
        with pytest.raises((TypeError, AttributeError)):
            TrendIndicators.t3(None)

    def test_t3_parameter_validation(self, sample_data):
        """Test T3 parameter validation"""
        # Test with different parameters
        for length in [3, 5, 10]:
            for a in [0.5, 0.7, 0.9]:
                result = TrendIndicators.t3(
                    sample_data,
                    length=length,
                    a=a
                )

                assert isinstance(result, pd.Series)
                assert len(result) == len(sample_data)

    def test_t3_all_parameters(self, sample_data):
        """Test T3 with all custom parameters"""
        length = 5
        a = 0.7

        result = TrendIndicators.t3(
            sample_data,
            length=length,
            a=a
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_t3_with_pandas_ta_fallback(self, sample_data):
        """Test T3 pandas-ta fallback behavior"""
        # This should use pandas-ta directly
        our_result = TrendIndicators.t3(
            sample_data
        )

        ta_result = ta.t3(
            sample_data
        )

        # Compare non-NaN values
        mask = our_result.notna() & ta_result.notna()
        pd.testing.assert_series_equal(
            our_result[mask],
            ta_result[mask],
            check_names=False,
            atol=1e-10
        )

    def test_t3_smoothing_effect(self, sample_data):
        """Test T3 smoothing effect with different parameters"""
        result_smooth = TrendIndicators.t3(
            sample_data,
            length=10,
            a=0.8  # Higher smoothing factor
        )

        result_less_smooth = TrendIndicators.t3(
            sample_data,
            length=5,
            a=0.5  # Lower smoothing factor
        )

        # Both should be series
        assert isinstance(result_smooth, pd.Series)
        assert isinstance(result_less_smooth, pd.Series)

    def test_t3_calculation_stability(self, sample_data):
        """Test T3 calculation is stable and consistent"""
        result1 = TrendIndicators.t3(
            sample_data,
            length=5,
            a=0.7
        )

        result2 = TrendIndicators.t3(
            sample_data,
            length=5,
            a=0.7
        )

        # Should be identical for same inputs
        pd.testing.assert_series_equal(result1, result2)

    def test_t3_default_parameters(self, sample_data):
        """Test T3 with default parameters"""
        result = TrendIndicators.t3(sample_data)

        # T3 should use length=5, a=0.7 as defaults
        expected = TrendIndicators.t3(sample_data, length=5, a=0.7)

        pd.testing.assert_series_equal(result, expected)

    def test_t3_a_parameter_range(self, sample_data):
        """Test T3 with various a parameter values"""
        for a_val in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            result = TrendIndicators.t3(
                sample_data,
                a=a_val
            )
            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_data)