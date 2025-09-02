#!/usr/bin/env python3
"""
Quick test script for T3 indicator with vfactor parameter
"""

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
import pandas as pd
import sys

def test_t3():
    print("Testing T3 with vfactor parameter...", flush=True)

    # Create test data
    data = pd.Series([100, 102, 98, 105, 103, 107, 111, 109, 115, 117], name='close')
    df = pd.DataFrame({'close': data})
    print(f"Data: {data.tolist()}", flush=True)

    # Test T3 with vfactor parameter
    service = TechnicalIndicatorService()
    try:
        result = service.calculate_indicator(df, 'T3', {'length': 5, 'vfactor': 0.8})
        print(f"T3 result type: {type(result)}", flush=True)

        if result is not None:
            if hasattr(result, '__len__'):
                print(f"T3 result length: {len(result)}", flush=True)
                if len(result) > 5:
                    print(f"T3 first 5 values: {result[:5].tolist() if hasattr(result[:5], 'tolist') else result[:5]}", flush=True)
                else:
                    print(f"T3 values: {result.tolist() if hasattr(result, 'tolist') else result}", flush=True)
            print("✓ T3 SUCCESS: vfactor parameter accepted and processed", flush=True)
            return True
        else:
            print("✗ T3 FAILED: returned None", flush=True)
            return False
    except Exception as e:
        print(f"✗ T3 FAILED: {str(e)}", flush=True)
        return False

if __name__ == "__main__":
    success = test_t3()
    sys.exit(0 if success else 1)