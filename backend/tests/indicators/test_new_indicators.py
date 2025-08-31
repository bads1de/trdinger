"""
新しい指標のテスト
"""
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
import pandas as pd
import numpy as np
import pytest

def test_new_indicators():
    """新しい指標をテスト"""
    # Test data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 50)
    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': [price * (1 + np.random.normal(0, 0.005)) for price in close_prices[:50]],
        'high': [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices[:50]],
        'low': [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices[:50]],
        'close': close_prices[:50],
        'volume': np.random.uniform(1000000, 10000000, 50)
    })

    service = TechnicalIndicatorService()

    # Test indicators
    indicators_to_test = [
        ('HL2', {}),
        ('OHLC4', {}),
        ('CCI', {'period': 14}),
        ('AOBV', {'fast': 5, 'slow': 10, 'max_lookback': 2, 'min_lookback': 2, 'mamode': 'ema'}),
        ('HWC', {'na': 0.2, 'nb': 0.1, 'nc': 3.0, 'nd': 0.3, 'scalar': 2.0})
    ]

    for indicator_name, params in indicators_to_test:
        print(f'Testing {indicator_name}...')
        try:
            result = service.calculate_indicator(df, indicator_name, params)
            if result is not None:
                print(f'{indicator_name}: SUCCESS - {type(result)}')
            else:
                print(f'{indicator_name}: FAILED - None result')
        except Exception as e:
            print(f'{indicator_name}: ERROR - {e}')

def test_cci_error_handling():
    """CCI関数のエラーハンドリングテスト"""
    print("=== CCI Error Handling Tests ===")

    # テストデータ生成
    np.random.seed(42)
    valid_high = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
    valid_low = pd.Series([98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
    valid_close = pd.Series([99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116])

    # テストケース1: 正常データ
    print("Test 1: Valid data")
    try:
        result = MomentumIndicators.cci(valid_high, valid_low, valid_close, length=14)
        print(f"[PASS] SUCCESS - Normal data: result length={len(result)}")
    except Exception as e:
        print(f"[ERROR] Normal data: {e}")

    # テストケース2: 不足データ
    print("\nTest 2: Insufficient data")
    short_high = pd.Series([100, 101, 102])
    short_low = pd.Series([98, 99, 100])
    short_close = pd.Series([99, 100, 101])
    try:
        result = MomentumIndicators.cci(short_high, short_low, short_close, length=14)
        print(f"[FAIL] UNEXPECTED SUCCESS - Should have failed for insufficient data")
    except ValueError as e:
        print(f"[PASS] EXPECTED ERROR - Insufficient data: {e}")
    except Exception as e:
        print(f"[FAIL] UNEXPECTED ERROR - Insufficient data: {e}")

    # テストケース3: NaN値を含むデータ
    print("\nTest 3: NaN values")
    nan_high = valid_high.copy()
    nan_high.iloc[5] = np.nan
    try:
        result = MomentumIndicators.cci(nan_high, valid_low, valid_close, length=14)
        print(f"? SUCCESS with NaN - Check result: has_nan={pd.isna(result).any()}")
    except Exception as e:
        print(f"[WARN] EXPECTED/HANDLED ERROR - NaN values: {e}")

    # テストケース4: 全NaNデータ
    print("\nTest 4: All NaN data")
    try:
        all_nan = pd.Series([np.nan] * 20)
        result = MomentumIndicators.cci(all_nan, all_nan, all_nan, length=14)
        print("[FAIL] UNEXPECTED SUCCESS - Should have failed for all NaN")
    except ValueError as e:
        print(f"[PASS] EXPECTED ERROR - All NaN: {e}")
    except Exception as e:
        print(f"[FAIL] UNEXPECTED ERROR - All NaN: {e}")

    # テストケース5: 不正なlength
    print("\nTest 5: Invalid length")
    try:
        result = MomentumIndicators.cci(valid_high, valid_low, valid_close, length=0)
        print("[FAIL] UNEXPECTED SUCCESS - Should have failed for invalid length")
    except ValueError as e:
        print(f"[PASS] EXPECTED ERROR - Invalid length: {e}")
    except Exception as e:
        print(f"[FAIL] UNEXPECTED ERROR - Invalid length: {e}")

    # テストケース6: 負の価格
    print("\nTest 6: Negative prices")
    neg_price = valid_close.copy()
    neg_price.iloc[5] = -10
    try:
        result = MomentumIndicators.cci(valid_high, valid_low, neg_price, length=14)
        print(f"? SUCCESS with negative price - Check result: has_nan={pd.isna(result).any()}")
    except Exception as e:
        print(f"[WARN] EXPECTED/HANDLED ERROR - Negative price: {e}")

    # テストケース7: 不正なOHLC関係
    print("\nTest 7: Invalid OHLC relationship (high < close)")
    invalid_high = valid_high.copy()
    invalid_high.iloc[5] = 95  # high < close
    try:
        result = MomentumIndicators.cci(invalid_high, valid_low, valid_close, length=14)
        print("[PASS] SUCCESS with corrected OHLC relationship")
    except Exception as e:
        print(f"[WARN] EXPECTED/HANDLED ERROR - Invalid OHLC: {e}")

    print("\n=== CCI Error Handling Tests Complete ===")

if __name__ == "__main__":
    test_new_indicators()
    test_cci_error_handling()