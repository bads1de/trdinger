#!/usr/bin/env python3
"""
STC修正テストスクリプト
"""

import pandas as pd
import numpy as np
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

def test_stc_fix():
    print("=== STC修正テスト開始 ===")

    # Create sample data
    np.random.seed(42)
    close_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120]
    df = pd.DataFrame({'close': close_prices})

    print(f"Test data shape: {df.shape}")
    print(f"Close prices: {close_prices}")

    # Initialize service
    service = TechnicalIndicatorService()
    print("Service initialized successfully")

    # Get supported indicators
    supported = service.get_supported_indicators()
    print(f"Total supported indicators: {len(supported)}")

    if 'STC' in supported:
        print("STC is in supported indicators")
    else:
        print("STC is NOT in supported indicators")
        return False

    # Test STC with different parameter combinations
    test_cases = [
        {'tclength': 10, 'fast': 23, 'slow': 50, 'factor': 0.5},
        {'tclength': 8, 'fast': 12, 'slow': 26, 'factor': 0.5},
        {},  # Default parameters
    ]

    for i, params in enumerate(test_cases):
        print(f"\n--- Test case {i+1}: {params} ---")
        try:
            result = service.calculate_indicator(df, 'STC', params)
            if result is not None:
                print("SUCCESS: STC calculation completed")
                print(f"Result shape: {result.shape}")
                print(f"Result type: {type(result)}")
                print(f"Sample values (last 5): {result[-5:]}")
                print(f"Value range: {result.min():.2f} to {result.max():.2f}")
                return True
            else:
                print("FAILED: Result is None")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    return False

if __name__ == "__main__":
    success = test_stc_fix()
    print(f"\n=== テスト結果: {'成功' if success else '失敗'} ===")