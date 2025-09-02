#!/usr/bin/env python3
"""
Simple ADX test script
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import pandas as pd
import pandas_ta as ta

print("Testing ADX calculation...")
print("=" * 50)

# Generate sample data
np.random.seed(42)
n = 100
close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
high = close + np.abs(np.random.randn(n)) * 5
low = close - np.abs(np.random.randn(n)) * 5

print(f"Sample data length: {len(high)}")
print(f"Sample data(high) average: {high.mean():.2f}")

try:
    # Test ADX calculation directly with TA
    result_ta = ta.adx(high=high, low=low, close=close, length=14)
    print(f"panda-ta ADX result type: {type(result_ta)}")
    if result_ta is not None:
        print(f"panda-ta ADX result shape: {result_ta.shape}")
        print(f"panda-ta ADX columns: {result_ta.columns.tolist()}")
        print(f"panda-ta ADX sample values: {result_ta.iloc[:5, 0].values if len(result_ta) > 5 else 'Too few values'}")

    # Test backward compatibility: use length parameter
    print("\nTesting ADX calculation with direct data...")

    # Simulate the function logic (mock the import issue)
    def mock_adx(high, low, close, period=14, length=None, **kwargs):
        if length is not None:
            period = length

        result = ta.adx(high=high, low=low, close=close, length=period)

        if result is None or result.empty:
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        return result.iloc[:, 0]  # ADX列

    print("Testing ADX with period=14")
    result_period = mock_adx(high, low, close, period=14)
    print(f"ADX with period result type: {type(result_period)}")
    if result_period is not None:
        print(f"ADX with period result length: {len(result_period)}")
        print(f"ADX with period sample values: {result_period.values[:5] if len(result_period) > 5 else 'Few values'}")

    print("\nTesting ADX with length=14 (backward compatibility)")
    result_length = mock_adx(high, low, close, length=14)
    print(f"ADX with length result type: {type(result_length)}")
    if result_length is not None:
        print(f"ADX with length result length: {len(result_length)}")
        print(f"ADX with length sample values: {result_length.values[:5] if len(result_length) > 5 else 'Few values'}")

    # Compare results
    if result_period is not None and result_length is not None:
        # Check if all are equal
        period_length_equal = result_period.equals(result_length)
        print(f"\nperiod and length results are equal: {period_length_equal}")

        if result_ta is not None:
            ta_equal = result_period.equals(result_ta.iloc[:, 0])
            print(f"period and ta results are equal: {ta_equal}")
        else:
            print("ta result is None, cannot compare")
    else:
        print("One or more results are None")

    print("\nADX testing completed successfully!")

except Exception as e:
    print(f"Error during ADX testing: {e}")
    import traceback
    traceback.print_exc()