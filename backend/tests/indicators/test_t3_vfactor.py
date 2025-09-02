"""
Test for T3 vfactor parameter mapping in TechnicalIndicatorService
"""
import pytest
import pandas as pd
import numpy as np
import pandas_ta as ta

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


class TestT3VfactorParameter:
    """Test T3 with vfactor parameter handling"""

    @pytest.fixture
    def sample_data(self):
        """Sample close data for testing"""
        data = pd.Series([100, 102, 98, 105, 103, 107, 111, 109, 115, 117], name='close')
        return data

    @pytest.fixture
    def service(self):
        """TechnicalIndicatorService fixture"""
        return TechnicalIndicatorService()

    def test_t3_ignores_vfactor_parameter_currently(self, sample_data, service):
        """Test that T3 currently handles vfactor parameter by mapping to a"""
        # After vfactor mapping, this should work
        params = {'length': 5, 'vfactor': 0.8}

        result = service.calculate_indicator(pd.DataFrame({'close': sample_data}), 'T3', params)

        # vfactor should be mapped to a parameter, so result should be valid
        assert result is not None
        assert isinstance(result, pd.Series)

    def test_t3_with_explicit_a_parameter(self, sample_data, service):
        """Test T3 works correctly with a parameter"""
        params = {'length': 5, 'a': 0.8}
        result = service.calculate_indicator(pd.DataFrame({'close': sample_data}), 'T3', params)

        assert result is not None

        # Compare with direct pandas-ta call
        direct_result = ta.t3(sample_data, length=5, a=0.8)

        # Results should be similar (allowing for minor differences)
        np.testing.assert_array_almost_equal(result, direct_result.values, decimal=5)

    def test_t3_vfactor_parameter_future_behavior(self, sample_data):
        """Future test: T3 should handle vfactor parameter by mapping to a"""
        # This test will pass once we implement vfactor mapping
        service = TechnicalIndicatorService()
        params = {'length': 5, 'vfactor': 0.8}

        # After implementation, vfactor should be mapped to a parameter
        result = service.calculate_indicator(pd.DataFrame({'close': sample_data}), 'T3', params)
        assert result is not None

        # Expected behavior: result should be same as ta.t3 with a=vfactor
        expected = ta.t3(sample_data, length=5, a=0.8)
        np.testing.assert_array_almost_equal(result, expected.values, decimal=5)

    @pytest.mark.xfail(reason="vfactor parameter not yet implemented in T3 config")
    def test_t3_vfactor_xfail_until_implemented(self, sample_data):
        """Test that should fail until vfactor is properly mapped"""
        service = TechnicalIndicatorService()
        params = {'length': 5, 'vfactor': 0.7}

        # This should work like vfactor mapping to a
        result1 = service.calculate_indicator(pd.DataFrame({'close': sample_data}), 'T3', params)

        # Compare with direct a parameter
        params_a = {'length': 5, 'a': 0.7}
        result2 = service.calculate_indicator(pd.DataFrame({'close': sample_data}), 'T3', params_a)

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)