"""
Tests for newly implemented failed indicators: VP, CWMA, VAR, CV, IRM
"""
import pytest
import pandas as pd
import numpy as np
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


@pytest.fixture
def indicator_service():
    return TechnicalIndicatorService()


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 100)
    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': [price * (1 + np.random.normal(0, 0.005)) for price in close_prices],
        'high': [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices],
        'low': [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices],
        'close': close_prices,
        'volume': np.random.uniform(1000000, 10000000, 100)
    })
    return df


def test_cwma_indicator(indicator_service, sample_data):
    """CWMA indicator test"""
    result = indicator_service.calculate_indicator(sample_data, "CWMA", {"length": 10})

    assert result is not None
    assert isinstance(result, (np.ndarray, pd.Series))
    assert len(result) == len(sample_data)
    # CWMA should be defined for the last values
    if isinstance(result, pd.Series):
        assert not np.isnan(result.iloc[-1])
    else:
        assert not np.isnan(result[-1])
    print("CWMA test passed")


def test_var_indicator(indicator_service, sample_data):
    """VAR indicator test"""
    result = indicator_service.calculate_indicator(sample_data, "VAR", {"length": 14})

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result) == len(sample_data)
    # Variance should be positive for the last values
    assert not np.isnan(result[-1])
    print("VAR test passed")


def test_cv_indicator(indicator_service, sample_data):
    """CV indicator test"""
    result = indicator_service.calculate_indicator(sample_data, "CV", {"length": 14})

    assert result is not None
    assert isinstance(result, (np.ndarray, pd.Series))
    assert len(result) == len(sample_data)
    # CV should be defined for the last values
    if isinstance(result, pd.Series):
        assert not np.isnan(result.iloc[-1])
    else:
        assert not np.isnan(result[-1])
    print("CV test passed")


def test_irm_indicator(indicator_service, sample_data):
    """IRM indicator test"""
    result = indicator_service.calculate_indicator(sample_data, "IRM", {"length": 14})

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result) == len(sample_data)
    # IRM should be defined for the last values
    assert not np.isnan(result[-1])
    print("IRM test passed")


def test_vp_improved_indicator(indicator_service, sample_data):
    """VP indicator improved test"""
    result = indicator_service.calculate_indicator(sample_data, "VP", {"width": 10})

    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 6  # VP returns 6 arrays (low, mean, high, pos_vol, neg_vol, total_vol)
    # VP returns volume profile bins, length depends on number of bins
    for arr in result:
        assert isinstance(arr, (np.ndarray, pd.Series))
        assert len(arr) > 0  # At least some bins
    print("VP improved test passed")


if __name__ == "__main__":
    import sys
    sys.path.append('backend')

    service = TechnicalIndicatorService()
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 100)
    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)

    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': [price * (1 + np.random.normal(0, 0.005)) for price in close_prices],
        'high': [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices],
        'low': [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices],
        'close': close_prices,
        'volume': np.random.uniform(1000000, 10000000, 100)
    })

    # Test each indicator
    print("Testing CWMA...")
    test_cwma_indicator(service, sample_df)

    print("Testing VAR...")
    test_var_indicator(service, sample_df)

    print("Testing CV...")
    test_cv_indicator(service, sample_df)

    print("Testing IRM...")
    test_irm_indicator(service, sample_df)

    print("Testing VP...")
    test_vp_improved_indicator(service, sample_df)

    print("All tests passed!")