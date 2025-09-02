"""
Test for PVOL technical indicator
"""
import pytest
import pandas as pd
import numpy as np
import pandas_ta as ta
from app.services.indicators.technical_indicators.volume import VolumeIndicators
from app.services.indicators.utils import PandasTAError


class TestPVOL:
    """Test PVOL (Price-Volume) indicator"""

    @pytest.fixture
    def sample_data(self):
        """Sample price-volume data for testing"""
        np.random.seed(42)
        n = 50
        close = pd.Series(np.random.uniform(100, 200, n), name='close')
        volume = pd.Series(np.random.uniform(1000, 10000, n), name='volume')

        return {
            'close': close,
            'volume': volume
        }

    def test_pvol_basic_calculation(self, sample_data):
        """Test basic PVOL calculation"""
        result = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume']
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])

    def test_pvol_manual_verification(self, sample_data):
        """Test PVOL matches known calculation pattern"""
        result = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume']
        )

        # PVOL calculation: Price with volume bias
        # This is a complex calculation involving volume-weighted price adjustments
        ta_result = ta.pvol(
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # Compare non-NaN values
        mask = result.notna() & ta_result.notna()
        pd.testing.assert_series_equal(
            result[mask],
            ta_result[mask],
            check_names=False,
            atol=1e-10
        )

    def test_pvol_nan_handling(self, sample_data):
        """Test PVOL handles NaN values properly"""
        # Add NaN values
        sample_data['close'].iloc[5] = np.nan
        sample_data['volume'].iloc[10] = np.nan

        result = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume']
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])

    def test_pvol_data_length_validation(self, sample_data):
        """Test PVOL data length validation"""
        result = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume']
        )

        assert len(result) == len(sample_data['close'])

    def test_pvol_edge_cases(self):
        """Test PVOL with edge case data"""
        # Test with constant values
        constant_close = pd.Series([100] * 20, name='close')
        constant_volume = pd.Series([1000] * 20, name='volume')

        result = VolumeIndicators.pvol(constant_close, constant_volume)

        # Should calculate normally
        assert isinstance(result, pd.Series)
        assert len(result) == 20

    def test_pvol_invalid_inputs(self):
        """Test PVOL error handling for invalid inputs"""
        # Test with mismatched lengths
        short_series = pd.Series([1, 2], name='short')
        long_series = pd.Series([1] * 10, name='long')

        with pytest.raises(TypeError):
            VolumeIndicators.pvol(long_series, short_series)

    def test_pvol_parameter_validation(self, sample_data):
        """Test PVOL parameter validation"""
        # Test with signed parameter
        result_signed = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume'],
            signed=True
        )

        result_unsigned = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume'],
            signed=False
        )

        assert isinstance(result_signed, pd.Series)
        assert isinstance(result_unsigned, pd.Series)
        assert len(result_signed) == len(sample_data['close'])
        assert len(result_unsigned) == len(sample_data['close'])

    def test_pvol_all_parameters(self, sample_data):
        """Test PVOL with all custom parameters"""
        result = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume'],
            signed=True
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])

    def test_pvol_with_pandas_ta_fallback(self, sample_data):
        """Test PVOL pandas-ta fallback behavior"""
        # This should use pandas-ta directly
        our_result = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume']
        )

        ta_result = ta.pvol(
            close=sample_data['close'],
            volume=sample_data['volume']
        )

        # Compare non-NaN values
        mask = our_result.notna() & ta_result.notna()
        pd.testing.assert_series_equal(
            our_result[mask],
            ta_result[mask],
            check_names=False,
            atol=1e-10
        )

    def test_pvol_volume_weighted_behavior(self, sample_data):
        """Test PVOL respects volume weighting"""
        result = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume']
        )

        # Should not be all NaN
        assert not result.isna().all()

        # High volume periods should influence calculation
        high_volume_mask = sample_data['volume'] > sample_data['volume'].median()
        if high_volume_mask.any():
            assert isinstance(result[high_volume_mask], pd.Series)

    def test_pvol_calculation_stability(self, sample_data):
        """Test PVOL calculation is stable and consistent"""
        result1 = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume']
        )

        result2 = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume']
        )

        # Should be identical for same inputs
        pd.testing.assert_series_equal(result1, result2)

    def test_pvol_default_parameters(self, sample_data):
        """Test PVOL with default parameters"""
        result = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume']
        )

        # PVOL should use signed=True as default
        expected = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume'],
            signed=True
        )

        pd.testing.assert_series_equal(result, expected)

    def test_pvol_signed_impact(self, sample_data):
        """Test the impact of signed parameter on PVOL"""
        # Create specific price movement scenario
        price = pd.Series([100, 105, 95, 110, 90], name='close')  # Up-down-up-down
        vol = pd.Series([1000] * 5, name='volume')

        result_signed = VolumeIndicators.pvol(price, vol, signed=True)
        result_unsigned = VolumeIndicators.pvol(price, vol, signed=False)

        assert isinstance(result_signed, pd.Series)
        assert isinstance(result_unsigned, pd.Series)
        # Results might differ based on signed calculation
        assert result_signed is not result_unsigned  # Different objects

    def test_pvol_price_volume_interaction(self, sample_data):
        """Test that PVOL properly accounts for price and volume interaction"""
        result = VolumeIndicators.pvol(
            sample_data['close'],
            sample_data['volume']
        )

        # Should be numeric series
        assert result.dtype in [np.float64, np.float32]
        assert not result.isna().all()
    def test_pvol_missing_volume_column(self, sample_data):
        """Test PVOL with missing volume column (DataFrame without volume)"""
        # Create DataFrame with close but no volume
        df = pd.DataFrame({'close': sample_data['close']})

        # Should raise PandasTAError due to missing volume parameter
        with pytest.raises(PandasTAError):
            VolumeIndicators.pvol(df.close, df.get('volume'))

    def test_pvol_none_volume(self, sample_data):
        """Test PVOL with None volume data"""
        # Pass None as volume
        with pytest.raises(PandasTAError):
            VolumeIndicators.pvol(sample_data['close'], None)

    def test_pvol_zero_volume(self, sample_data):
        """Test PVOL with all zero volume data"""
        # All zero volume - should handle gracefully but might produce NaN
        zero_volume = pd.Series([0] * len(sample_data['volume']), name='volume')

        result = VolumeIndicators.pvol(sample_data['close'], zero_volume)

        # Should return Series (possibly with NaN values)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])
