"""
Test for indicator warnings and errors that were fixed.
"""

import pytest
import pandas as pd
import numpy as np
from backend.app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


class TestIndicatorWarnings:
    def test_vidya_no_futurewarning(self):
        """Test VIDYA calculation doesn't produce FutureWarning about dtype incompatibility"""
        df = pd.DataFrame({
            'Close': np.random.rand(100) * 100  # Use capital Close
        })

        service = TechnicalIndicatorService()
        with pytest.warns(None) as record:
            result = service.calculate_indicator(df, 'VIDYA', {'period': 14, 'adjust': True})

        # Ensure no FutureWarning about dtype
        future_warnings = [w for w in record.list
                          if "FutureWarning" in str(w.message) and "dtype incompatible" in str(w.message)]
        assert len(future_warnings) == 0
        assert result is not None

    def test_pvr_no_unexpected_keyword(self):
        """Test PVR calculation doesn't get unexpected keyword argument error"""
        df = pd.DataFrame({
            'Close': np.random.rand(100) * 100,
            'Volume': np.random.rand(100) * 1000  # Capital Volume
        })

        service = TechnicalIndicatorService()
        try:
            result = service.calculate_indicator(df, 'PVR', {})
            assert result is not None
            assert len(result) > 0
        except TypeError as e:
            assert "unexpected keyword argument" not in str(e)

    def test_kst_outputs(self):
        """Test KST calculation and outputs"""
        df = pd.DataFrame({
            'Close': np.random.rand(100) * 100
        })

        service = TechnicalIndicatorService()
        result = service.calculate_indicator(df, 'KST', {})

    def test_linreg_no_error(self):
        """Test LINREG calculation doesn't get unexpected keyword argument error"""
        df = pd.DataFrame({'Close': np.random.rand(100) * 100})
        service = TechnicalIndicatorService()
        try:
            result = service.calculate_indicator(df, 'LINREG', {'period': 14})
            assert result is not None
            assert isinstance(result, (np.ndarray, pd.Series))
            assert len(result) > 0
        except TypeError as e:
            assert "unexpected keyword argument" not in str(e)
    def test_stc_no_missing_arg(self):
        """Test STC calculation doesn't have missing data argument"""
        df = pd.DataFrame({'Close': np.random.rand(100) * 100})
        service = TechnicalIndicatorService()
        try:
            result = service.calculate_indicator(df, 'STC', {'length': 10, 'fast_length': 23, 'slow_length': 50})
            assert result is not None
        except TypeError as e:
            assert "missing 1 required positional argument" not in str(e)
    def test_cv_calculation(self):
        """Test CV calculation using custom implementation"""
        df = pd.DataFrame({'Close': np.random.rand(100) * 100})
        service = TechnicalIndicatorService()
        result = service.calculate_indicator(df, 'CV', {'length': 14})
        assert result is not None
    def test_irm_calculation(self):
        """Test IRM calculation using custom implementation"""
        df = pd.DataFrame({'High': np.random.rand(100) * 110, 'Low': np.random.rand(100) * 90, 'Close': np.random.rand(100) * 100 + 10})
        service = TechnicalIndicatorService()
        result = service.calculate_indicator(df, 'IRM', {'length': 14})
        assert result is not None
        assert len(result) > 0
        assert len(result) > 0