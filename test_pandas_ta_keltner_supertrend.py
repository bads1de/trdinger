"""
pandas-ta KELTNERとSUPERTREND指標のテストスクリプト
"""

import pandas as pd
import numpy as np
import pandas_ta as ta

# サンプルデータ生成
np.random.seed(42)
n = 100
close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), name='close')
high = pd.Series(close + np.random.rand(n) * 2, name='high')
low = pd.Series(close - np.random.rand(n) * 2, name='low')

print("pandas-ta version:", ta.__version__)
print("Data length:", len(close))

# KELTNER Channel test
print("\n=== KELTNER Channel Test ===")
try:
    # volatility.pyで使われているパラメータ
    period = 20
    scalar = 2.0

    print(f"Testing ta.kc with length={period}, scalar={scalar}")
    kc_result = ta.kc(high=high, low=low, close=close, length=period, scalar=scalar)
    print("KELTNER Success!")
    print("Columns:", kc_result.columns.tolist())
    print("Result shape:", kc_result.shape)
    print("First few rows:")
    print(kc_result.head())

    # volatility.pyで使われている呼び出し（windowパラメータ）
    try:
        print(f"\nTesting ta.kc with window={period}, scalar={scalar} (as in volatility.py)")
        kc_result_window = ta.kc(high=high, low=low, close=close, window=period, scalar=scalar)
        print("Window parameter works!")
    except Exception as e:
        print(f"Window parameter failed: {e}")

except Exception as e:
    print(f"KELTNER failed: {e}")

# SUPERTREND test
print("\n=== SUPERTREND Test ===")
try:
    # volatility.pyで使われているパラメータ
    period = 10
    multiplier = 3.0

    print(f"Testing ta.supertrend with length={period}, multiplier={multiplier}")
    st_result = ta.supertrend(high=high, low=low, close=close, length=period, multiplier=multiplier)
    print("SUPERTREND Success!")
    print("Columns:", st_result.columns.tolist())
    print("Result shape:", st_result.shape)
    print("First few rows:")
    print(st_result.head())

    # volatility.pyで使われている呼び出し（windowパラメータ）
    try:
        print(f"\nTesting ta.supertrend with window={period}, multiplier={multiplier} (as in volatility.py)")
        st_result_window = ta.supertrend(high=high, low=low, close=close, window=period, multiplier=multiplier)
        print("Window parameter works!")
    except Exception as e:
        print(f"Window parameter failed: {e}")

except Exception as e:
    print(f"SUPERTREND failed: {e}")

print("\n=== Parameter Mapping Issues ===")
print("KELTNER: volatility.py uses 'window' but pandas-ta uses 'length'")
print("SUPERTREND: volatility.py uses 'window' but pandas-ta uses 'length'")