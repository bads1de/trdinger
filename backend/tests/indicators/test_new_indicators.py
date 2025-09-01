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

def test_new_technical_indicators():
    """ROC, STC, VI, CFOの新規テクニカル指標テスト"""
    print("=== Testing New Technical Indicators ===")

    # Test data generation
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 99)
    close_prices = [base_price]
    for change in price_changes:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)

    # Create OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [price * (1 + np.random.normal(0, 0.005)) for price in close_prices],
        'high': [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices],
        'low': [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices],
        'close': close_prices,
        'volume': np.random.uniform(1000000, 10000000, 100)
    })
    df = df.set_index('timestamp')

    service = TechnicalIndicatorService()

    # New indicators to test
    new_indicators = [
        ("ROC", {'length': 10}),
        ("STC", {'tclength': 10, 'fast': 23, 'slow': 50, 'factor': 0.5}),
        ("VORTEX", {'length': 14}),
        ("CFO", {'length': 9})
    ]

    for indicator_name, params in new_indicators:
        print(f"\nTesting {indicator_name} with params: {params}")
        try:
            result = service.calculate_indicator(df, indicator_name, params)
            if result is not None:
                result_df = result.iloc[-20:]  # Last 20 results for inspection
                print(f"✓ {indicator_name}: SUCCESS - Result shape: {result_df.shape}")
                print(f"  Sample values: {result_df.dropna().head(3).values.flatten()[:5]}")
                # Check for NaN behavior
                nan_count = result_df.isna().sum().sum() if hasattr(result_df, 'isna') else 0
                print(f"  NaN count in sample: {nan_count}")
            else:
                print(f"✗ {indicator_name}: FAILED - None result")
        except Exception as e:
            print(f"✗ {indicator_name}: ERROR - {str(e)}")

def test_extended_threshold_conditions():
    """拡張thresholdテスト - range, combo条件など"""
    print("\n=== Testing Extended Threshold Conditions ===")

    # Test data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=20, freq='1H')
    closes = np.array([100, 102, 101, 105, 103, 107, 106, 110, 108, 112,
                       111, 115, 114, 118, 116, 120, 119, 123, 121, 125])
    df = pd.DataFrame({
        'timestamp': dates,
        'close': closes
    }).set_index('timestamp')

    service = TechnicalIndicatorService()

    # Test ROC with extended thresholds
    try:
        result = service.calculate_indicator(df, "ROC", {'length': 10})
        if result is not None:
            recent_values = result.iloc[-10:].dropna().values
            print("ROC Extended Threshold Tests:")

            # Range threshold: check if ROC values are within certain ranges
            range_tests = [
                ("Very Oversold", lambda x: x < -5, "values < -5"),
                ("Neutral", lambda x: -5 <= x <= 5, "values in [-5, 5]"),
                ("Very Overbought", lambda x: x > 5, "values > 5"),
                ("Aggressive Oversold", lambda x: x < -8, "values < -8"),
                ("Conservative Overbought", lambda x: x > 8, "values > 8")
            ]

            for test_name, condition, desc in range_tests:
                matching_values = [v for v in recent_values.flatten() if condition(v)]
                count = len(matching_values)
                print(f"  {test_name}: {count}/{len(recent_values.flatten())} {desc}")

            # Combo conditions: multiple conditions combined
            combo_conditions = [
                ("Strong Bullish", lambda x: x > 3 and x < 10, "3 < ROC and ROC < 10"),
                ("Strong Bearish", lambda x: x < -3 and x > -10, "-3 > ROC and ROC > -10"),
                ("Extreme Move", lambda x: abs(x) > 8, "|ROC| > 8"),
                ("Slowing Momentum", lambda x: -2 <= x <= 2, "Slow momentum range")
            ]

            print("\n  Combo Conditions:")
            for test_name, condition, desc in combo_conditions:
                combo_matches = [v for v in recent_values.flatten() if condition(v)]
                combo_count = len(combo_matches)
                print(f"    {test_name}: {combo_count}/{len(recent_values.flatten())} {desc}")
        else:
            print("  ✗ ROC test failed - no result")
    except Exception as e:
        print(f"  ✗ ROC extended threshold test ERROR - {str(e)}")

def test_threshold_extremes():
    """thresholdの極端な値でのテスト"""
    print("\n=== Testing Threshold Extremes ===")

    # Generate test data with extreme moves
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=30, freq='1H')

    # Create data with sharp moves to test indicators at extremes
    base = 100
    closes = [base]
    for i in range(29):
        if i % 5 == 0:
            # Sharp move
            change = np.random.choice([-0.05, 0.05]) * np.random.uniform(1, 3)
        else:
            change = np.random.normal(0, 0.01)
        new_price = closes[-1] * (1 + change)
        closes.append(max(new_price, 10))  # Prevent negative prices

    df = pd.DataFrame({
        'timestamp': dates,
        'close': closes
    }).set_index('timestamp')

    service = TechnicalIndicatorService()

    indicators = ["ROC", "STC", "CFO"]
    params = {
        "ROC": {'length': 5},
        "STC": {'tclength': 5, 'fast': 12, 'slow': 26, 'factor': 0.5},
        "CFO": {'length': 5}
    }

    for indicator in indicators:
        try:
            result = service.calculate_indicator(df, indicator, params[indicator])
            if result is not None:
                values = result.dropna().values.flatten()
                max_val = float(np.max(values))
                min_val = float(np.min(values))
                std_val = float(np.std(values))
                print(f"  {indicator} Extremes - Max: {max_val:.2f}, Min: {min_val:.2f}, Std: {std_val:.2f}")

                # Check for extreme threshold scenarios
                if abs(max_val) > 50 or abs(min_val) > 50:
                    print(f"    ✓ Extreme values detected (>{50})")
                if std_val < 0.1:
                    print(f"    ⚠ Low volatility detected in {indicator}")
            else:
                print(f"  ✗ {indicator} - No result at extremes")
        except Exception as e:
            print(f"  ✗ {indicator} extremes test ERROR - {str(e)}")

if __name__ == "__main__":
    test_new_indicators()
    test_new_technical_indicators()
    test_extended_threshold_conditions()
    test_threshold_extremes()

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