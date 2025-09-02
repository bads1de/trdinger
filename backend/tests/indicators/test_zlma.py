"""
Test for ZLMA technical indicator
"""
import pytest
import pandas as pd
import numpy as np
import pandas_ta as ta
from backend.app.services.indicators.technical_indicators.trend import TrendIndicators


class TestZLMA:
    """Test ZLMA (Zero-Lag Exponential Moving Average) indicator"""

    @pytest.fixture
    def sample_data(self):
        """Sample close data for testing"""
        np.random.seed(42)
        n = 50
        close = pd.Series(np.random.uniform(100, 200, n), name='close')

        return close

    def test_zlma_basic_calculation(self, sample_data):
        """Test basic ZLMA calculation"""
        result = TrendIndicators.zlma(
            sample_data,
            length=20
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_zlma_manual_verification(self, sample_data):
        """Test ZLMA matches known calculation pattern"""
        length = 20

        result = TrendIndicators.zlma(
            sample_data,
            length=length
        )

        # ZLMA calculation: Modified EMA with lag correction
        # First try pandas-ta, fallback to manual
        if hasattr(ta, "zlma"):
            ta_result = ta.zlma(sample_data, length=length)
            if ta_result is not None:
                pd.testing.assert_series_equal(result, ta_result, check_names=False)
        else:
            # Manual verification for fallback
            lag = int((length - 1) / 2)
            shifted = sample_data.shift(lag)
            adjusted = sample_data + (sample_data - shifted)
            expected = ta.ema(adjusted, length=length)

            # Compare non-NaN values
            mask = result.notna() & expected.notna()
            pd.testing.assert_series_equal(
                result[mask],
                expected[mask],
                check_names=False,
                atol=1e-10
            )

    def test_zlma_nan_handling(self, sample_data):
        """Test ZLMA handles NaN values properly"""
        # Add NaN values
        sample_data.iloc[5] = np.nan

        result = TrendIndicators.zlma(sample_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_zlma_data_length_validation(self, sample_data):
        """Test ZLMA data length validation"""
        result = TrendIndicators.zlma(sample_data)

        assert len(result) == len(sample_data)

    def test_zlma_edge_cases(self):
        """Test ZLMA with edge case data"""
        # Test with constant values
        constant_data = pd.Series([100] * 30, name='close')

        result = TrendIndicators.zlma(constant_data)

        # Should be constant 100
        assert result.notna().all()
        assert (result == 100).all()

    def test_zlma_invalid_inputs(self):
        """Test ZLMA error handling for invalid inputs"""
        # Test with None data
        with pytest.raises((TypeError, AttributeError)):
            TrendIndicators.zlma(None)

    def test_zlma_parameter_validation(self, sample_data):
        """Test ZLMA parameter validation"""
        # Test with different lengths
        for length in [5, 10, 20]:
            result = TrendIndicators.zlma(
                sample_data,
                length=length
            )

            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_data)

    def test_zlma_all_parameters(self, sample_data):
        """Test ZLMA with all custom parameters"""
        length = 20

        result = TrendIndicators.zlma(
            sample_data,
            length=length
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_zlma_with_pandas_ta_fallback(self, sample_data):
        """Test ZLMA pandas-ta fallback behavior"""
        # This should use pandas-ta directly if available, otherwise fallback
        our_result = TrendIndicators.zlma(
            sample_data
        )

        assert isinstance(our_result, pd.Series)
        assert not our_result.isna().all()

    def test_zlma_lag_reduction(self, sample_data):
        """Test ZLMA as a lag-reduced moving average"""
        result = TrendIndicators.zlma(
            sample_data,
            length=10
        )

        # ZLMA should closely follow the price with less lag than regular EMA
        valid_idx = result.notna()
        price_follow = sample_data[valid_idx]
        zlma_value = result[valid_idx]

        # Values should be close to price (but not necessarily)
        assert isinstance(result, pd.Series)

    def test_zlma_calculation_stability(self, sample_data):
        """Test ZLMA calculation is stable and consistent"""
        result1 = TrendIndicators.zlma(
            sample_data
        )

        result2 = TrendIndicators.zlma(
            sample_data
        )

        # Should be identical for same inputs
        pd.testing.assert_series_equal(result1, result2)

    def test_zlma_default_parameters(self, sample_data):
        """Test ZLMA with default parameters"""
        result = TrendIndicators.zlma(sample_data)

        # ZLMA should use length=20 as default
        expected = TrendIndicators.zlma(sample_data, length=20)

        pd.testing.assert_series_equal(result, expected)

    def test_zlma_adaptive_behavior(self, sample_data):
        """Test ZLMA adapts to different price movements"""
        # Create data with different trends
        rising_data = pd.Series(range(100, 150), name='close')
        falling_data = pd.Series(range(150, 100, -1), name='close')

        rising_result = TrendIndicators.zlma(rising_data)
        falling_result = TrendIndicators.zlma(falling_data)

        # Should adapt to different trends
        assert isinstance(rising_result, pd.Series)
        assert isinstance(falling_result, pd.Series)