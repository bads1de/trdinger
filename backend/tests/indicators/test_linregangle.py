#!/usr/bin/env python3
"""
LINREGSLOPE specific test following TDD approach
"""

import pandas as pd
import numpy as np
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


def test_linregslope_basic():
    """Basic LINREGSLOPE test"""
    print("=== Testing LINREGSLOPE Basic ===")

    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    np.random.seed(42)

    # Realistic price data
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 99)
    close_prices = [base_price]
    for change in price_changes:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(max(1, new_price))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': [price * (1 + np.random.normal(0, 0.002)) for price in close_prices],
        'high': [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices],
        'low': [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices],
        'close': close_prices,
        'volume': np.random.uniform(1000000, 10000000, 100)
    })

    service = TechnicalIndicatorService()

    print("Testing LINREGSLOPE calculation...")
    try:
        result = service.calculate_indicator(df, 'LINREGSLOPE', {'timeperiod': 14})

        if result is not None:
            if isinstance(result, np.ndarray):
                valid_count = np.sum(~np.isnan(result))
                print(f"SUCCESS: LINREGSLOPE returned {result.shape} array with {valid_count} valid values")
                return True
            else:
                print(f"SUCCESS: LINREGSLOPE returned {type(result)}")
                return True
        else:
            print("FAIL: LINREGSLOPE result is None")
            return False

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_linregslope_basic()

    if success:
        print("\nLINREGSLOPE test PASSED!")
        exit(0)
    else:
        print("\nLINREGSLOPE test FAILED!")
        exit(1)