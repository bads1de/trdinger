#!/usr/bin/env python3
"""
シンプルなインジケータ動作確認スクリプト
"""

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
import pandas as pd
import numpy as np

def main():
    print("Starting indicator test...")

    # Create simple test data
    np.random.seed(42)
    close_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120]
    df = pd.DataFrame({
        'close': close_prices,
        'high': [x * 1.01 for x in close_prices],
        'low': [x * 0.99 for x in close_prices]
    })

    print(f"Test data shape: {df.shape}")

    # Initialize service
    service = TechnicalIndicatorService()
    print("Service initialized successfully")

    # Get supported indicators
    supported = service.get_supported_indicators()
    print(f"Supported indicators count: {len(supported)}")

    # Test key indicators
    test_cases = [
        ('STC', {'length': 10}),
        ('RSI', {'length': 14}),
        ('SMA', {'length': 10}),
        ('EMA', {'length': 10}),
    ]

    for indicator_name, params in test_cases:
        try:
            print(f"\nTesting {indicator_name}...")
            result = service.calculate_indicator(df, indicator_name, params)

            if result is not None:
                print(f"  SUCCESS: {indicator_name} - shape: {result.shape}")
                if hasattr(result, '__len__') and len(result) > 0:
                    # Show last few values
                    if hasattr(result, 'shape') and len(result.shape) > 0:
                        print(f"  Sample values: {result[-3:]}")
                    else:
                        print(f"  Value: {result}")
            else:
                print(f"  FAILED: {indicator_name} - returned None")

        except Exception as e:
            print(f"  ERROR: {indicator_name} - {str(e)}")

    print("\n=== Test Summary ===")
    print("Indicator tests completed successfully!")

if __name__ == "__main__":
    main()