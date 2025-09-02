# Test for VWMA indicator

import pytest
import pandas as pd
import numpy as np
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

def create_sample_data():
    """Create sample data for VWMA test"""
    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    return df

def test_vwma_with_default_params():
    """VWMA indicator test with default parameters"""
    service = TechnicalIndicatorService()
    df = create_sample_data()
    result = service.calculate_indicator(df, 'VWMA', {})

    assert result is not None
    assert len(result) == len(df)
    # VWMA should not be all NaN
    assert not np.isnan(result).all()

def test_vwma_with_custom_params():
    """VWMA indicator test with custom parameters"""
    service = TechnicalIndicatorService()
    df = create_sample_data()
    params = {'length': 5}

    result = service.calculate_indicator(df, 'VWMA', params)

    assert result is not None
    assert len(result) == len(df)

    # First 4 should be NaN (5-1), fifth should have value
    for i in range(4):
        assert np.isnan(result[i])
    assert not np.isnan(result[4])