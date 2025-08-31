#!/usr/bin/env python3

"""直接テストスクリプト"""

import sys
import os
sys.path.append('.')

def test_normalize_params():
    print("Testing normalize_params method...")

    try:
        from app.services.indicators.config.indicator_config import IndicatorConfig, ParameterConfig

        # Test 1: RSI with period -> length conversion
        rsi_config = IndicatorConfig(
            indicator_name='RSI',
            parameters={
                'length': ParameterConfig('length', 14)
            }
        )
        params = {'period': 20}
        result = rsi_config.normalize_params(params)
        print(f"RSI Test: {params} -> {result}")
        assert result == {'length': 20}, f"Expected {{'length': 20}}, got {result}"

        # Test 2: SAR special case
        sar_config = IndicatorConfig(indicator_name='SAR')
        params_sar = {'acceleration': 0.02, 'maximum': 0.2}
        result_sar = sar_config.normalize_params(params_sar)
        print(f"SAR Test: {params_sar} -> {result_sar}")
        expected_sar = {'af': 0.02, 'max_af': 0.2}
        assert result_sar == expected_sar, f"Expected {expected_sar}, got {result_sar}"

        # Test 3: Volume indicator (no length)
        nvi_config = IndicatorConfig(indicator_name='NVI')
        params_nvi = {'close': 'Close', 'volume': 'Volume'}
        result_nvi = nvi_config.normalize_params(params_nvi)
        print(f"NVI Test: {params_nvi} -> {result_nvi}")
        expected_nvi = {'close': 'Close', 'volume': 'Volume'}
        assert result_nvi == expected_nvi, f"Expected {expected_nvi}, got {result_nvi}"

        print("All normalize_params tests PASSED")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_integration():
    print("Testing integration with orchestrator...")

    try:
        import pandas as pd
        import numpy as np
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        # サンプルデータ作成
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = {
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000, 10000, 100)
        }
        df = pd.DataFrame(data, index=dates)
        df.loc[df['Low'] > df['Open'], 'Low'] = df.loc[df['Low'] > df['Open'], 'Open']
        df.loc[df['High'] < df['Open'], 'High'] = df.loc[df['High'] < df['Open'], 'Open']

        service = TechnicalIndicatorService()

        # Test cases
        test_cases = [
            ('RSI', {'period': 14}),
            ('ADM', {'fast': 12, 'slow': 26, 'signal': 9}),
            ('STOCHF', {'fastk_period': 5, 'fastd_period': 3}),
        ]

        for indicator, params in test_cases:
            try:
                result = service.calculate_indicator(df, indicator, params)
                print(f"{indicator} ({params}): SUCCESS")
            except Exception as e:
                print(f"{indicator} ({params}): ERROR - {e}")
                return False

        print("Integration tests PASSED")

    except Exception as e:
        print(f"Integration ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = True
    success &= test_normalize_params()
    success &= test_integration()

    if success:
        print("\nAll tests PASSED! The normalize_params refactoring is working correctly.")
        sys.exit(0)
    else:
        print("\nSome tests FAILED! Please check the implementation.")
        sys.exit(1)