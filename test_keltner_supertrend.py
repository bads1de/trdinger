#!/usr/bin/env python3
"""
KELTNERとSUPERTRENDの手動テストスクリプト
"""

import numpy as np
import pandas as pd
from backend.app.services.indicators.technical_indicators.volatility import VolatilityIndicators

def create_test_data():
    """テストデータ作成"""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1H')

    base_price = 50000
    price_changes = np.random.normal(0, 0.01, n)

    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(max(1, new_price))

    high_prices = [price * (1 + abs(np.random.normal(0, 0.005))) for price in close_prices]
    low_prices = [price * (1 - abs(np.random.normal(0, 0.005))) for price in close_prices]

    return pd.DataFrame({
        'timestamp': dates,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })

def test_atr():
    """ATRテスト"""
    print("=== ATR TEST ===")
    df = create_test_data()
    high = df['high']
    low = df['low']
    close = df['close']

    print(f"Data length: {len(high)}")

    try:
        atr = VolatilityIndicators.atr(high, low, close, length=20)
        print("ATR calculation successful")
        print(f"ATR NaN count: {atr.isna().sum()}")
        print(f"ATR sample: {atr.head().values}")
        return True
    except Exception as e:
        print(f"ATR calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_keltner():
    """KELTNERテスト"""
    print("=== KELTNER TEST ===")
    df = create_test_data()
    high = df['high']
    low = df['low']
    close = df['close']

    print(f"Data length: {len(high)}")
    print(f"High sample: {high.head().values}")
    print(f"Low sample: {low.head().values}")
    print(f"Close sample: {close.head().values}")

    try:
        upper, middle, lower = VolatilityIndicators.keltner(high, low, close, period=20, scalar=2.0)
        print("KELTNER calculation successful")
        print(f"Upper NaN count: {upper.isna().sum()}")
        print(f"Middle NaN count: {middle.isna().sum()}")
        print(f"Lower NaN count: {lower.isna().sum()}")
        print(f"Upper sample: {upper.head().values}")
        print(f"Middle sample: {middle.head().values}")
        print(f"Lower sample: {lower.head().values}")
        return True
    except Exception as e:
        print(f"KELTNER calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_supertrend():
    """SUPERTRENDテスト"""
    print("\n=== SUPERTREND TEST ===")
    df = create_test_data()
    high = df['high']
    low = df['low']
    close = df['close']

    print(f"Data length: {len(high)}")

    try:
        lower, upper, direction = VolatilityIndicators.supertrend(high, low, close, period=10, multiplier=3.0)
        print("SUPERTREND calculation successful")
        print(f"Lower NaN count: {lower.isna().sum()}")
        print(f"Upper NaN count: {upper.isna().sum()}")
        print(f"Direction NaN count: {direction.isna().sum()}")
        print(f"Lower sample: {lower.head().values}")
        print(f"Upper sample: {upper.head().values}")
        print(f"Direction sample: {direction.head().values}")
        return True
    except Exception as e:
        print(f"SUPERTREND calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_atr()
    test_keltner()
    test_supertrend()