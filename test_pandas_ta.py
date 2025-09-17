#!/usr/bin/env python3
"""
pandas-taのテストスクリプト
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

def create_test_data():
    """テストデータ作成"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1H')

    base_price = 50000
    price_changes = np.random.normal(0, 0.01, n)

    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(max(1, new_price))

    high_prices = [price * (1 + abs(np.random.normal(0, 0.005))) for price in close_prices]
    low_prices = [price * (1 - abs(np.random.normal(0, 0.005))) for price in close_prices]

    return pd.Series(high_prices), pd.Series(low_prices), pd.Series(close_prices)

def test_pandas_ta_atr():
    """pandas-ta ATRテスト"""
    print("=== pandas-ta ATR TEST ===")
    high, low, close = create_test_data()

    print(f"Data length: {len(high)}")

    try:
        result = ta.atr(high=high, low=low, close=close, length=20)
        print("pandas-ta ATR successful")
        print(f"ATR NaN count: {result.isna().sum()}")
        print(f"ATR sample: {result.head().values}")
        return result
    except Exception as e:
        print(f"pandas-ta ATR failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_pandas_ta_kc():
    """pandas-ta KCテスト"""
    print("=== pandas-ta KC TEST ===")
    high, low, close = create_test_data()

    print(f"Data length: {len(high)}")

    try:
        result = ta.kc(high=high, low=low, close=close, length=20, scalar=2.0)
        print("pandas-ta KC successful")
        print(f"KC columns: {result.columns.tolist()}")
        print(f"KC NaN count: {result.isna().sum().sum()}")
        print(f"KC sample:\n{result.head()}")
        return result
    except Exception as e:
        print(f"pandas-ta KC failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_pandas_ta_supertrend():
    """pandas-ta Supertrendテスト"""
    print("=== pandas-ta Supertrend TEST ===")
    high, low, close = create_test_data()

    print(f"Data length: {len(high)}")

    try:
        result = ta.supertrend(high=high, low=low, close=close, length=10, multiplier=3.0)
        print("pandas-ta Supertrend successful")
        print(f"Supertrend columns: {result.columns.tolist()}")
        print(f"Supertrend NaN count: {result.isna().sum().sum()}")
        print(f"Supertrend sample:\n{result.head()}")
        return result
    except Exception as e:
        print(f"pandas-ta Supertrend failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_pandas_ta_atr()
    test_pandas_ta_kc()
    test_pandas_ta_supertrend()