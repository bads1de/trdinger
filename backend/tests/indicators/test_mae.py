#!/usr/bin/env python3
"""
MAE specific test following TDD approach
"""

import pandas as pd
import numpy as np
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


def test_mae_basic():
    """Basic MAE test"""
    print("=== Testing MAE Basic ===")

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

    # Create "predicted" prices (simulated)
    pred_prices = [price * (1 + np.random.normal(0, 0.01)) for price in close_prices]

    df = pd.DataFrame({
        'timestamp': dates,
        'open': [price * (1 + np.random.normal(0, 0.002)) for price in close_prices],
        'high': [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices],
        'low': [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices],
        'close': close_prices,
        'predicted': pred_prices,  # Add predicted for MAE calculation
        'volume': np.random.uniform(1000000, 10000000, 100)
    })

    service = TechnicalIndicatorService()

    print("Testing MAE calculation...")
    try:
        # Test with default parameters
        result = service.calculate_indicator(df, 'MAE', {})

        if result is not None:
            if isinstance(result, np.ndarray):
                valid_count = np.sum(~np.isnan(result))
                print(f"SUCCESS: MAE returned {result.shape} array with {valid_count} valid values")
                # Print sample values
                print(f"Sample MAE values: {result[~np.isnan(result)][:5]}")
                return True
            else:
                print(f"SUCCESS: MAE returned {type(result)}")
                return True
        else:
            print("FAIL: MAE result is None")
            return False

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_mae_with_params():
    """MAE test with parameters"""
    print("=== Testing MAE with Parameters ===")

    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
    np.random.seed(42)

    close_prices = np.random.normal(50000, 2000, 50)
    pred_prices = close_prices + np.random.normal(0, 500, 50)

    df = pd.DataFrame({
        'timestamp': dates,
        'close': close_prices,
        'predicted': pred_prices,
    })

    service = TechnicalIndicatorService()

    print("Testing MAE calculation with length parameter...")
    try:
        result = service.calculate_indicator(df, 'MAE', {'length': 10})

        if result is not None and isinstance(result, np.ndarray):
            print(f"SUCCESS: MAE with length=10 returned array of shape {result.shape}")
            return True
        else:
            print("FAIL: MAE with parameters failed")
            return False

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    success1 = test_mae_basic()
    success2 = test_mae_with_params()

    if success1 and success2:
        print("\nMAE tests PASSED!")
        exit(0)
    else:
        print("\nMAE tests FAILED!")
        exit(1)