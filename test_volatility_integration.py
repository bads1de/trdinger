"""
volatility.pyのKELTNERとSUPERTREND関数を直接テスト
"""

import pandas as pd
import numpy as np
import sys
import os

# backendディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.services.indicators.technical_indicators.volatility import VolatilityIndicators

# サンプルデータ生成
np.random.seed(42)
n = 100
close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), name='close')
high = pd.Series(close + np.random.rand(n) * 2, name='high')
low = pd.Series(close - np.random.rand(n) * 2, name='low')

print("Testing VolatilityIndicators.keltner")
print("Data length:", len(close))

try:
    upper, middle, lower = VolatilityIndicators.keltner(high=high, low=low, close=close, period=20, scalar=2.0)
    print("KELTNER Success!")
    print("Upper shape:", upper.shape if hasattr(upper, 'shape') else len(upper))
    print("Middle shape:", middle.shape if hasattr(middle, 'shape') else len(middle))
    print("Lower shape:", lower.shape if hasattr(lower, 'shape') else len(lower))
    print("Upper first 5:", upper.head())
    print("Middle first 5:", middle.head())
    print("Lower first 5:", lower.head())
except Exception as e:
    print(f"KELTNER Failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting VolatilityIndicators.supertrend")
try:
    lower_st, upper_st, direction = VolatilityIndicators.supertrend(high=high, low=low, close=close, period=10, multiplier=3.0)
    print("SUPERTREND Success!")
    print("Lower shape:", lower_st.shape if hasattr(lower_st, 'shape') else len(lower_st))
    print("Upper shape:", upper_st.shape if hasattr(upper_st, 'shape') else len(upper_st))
    print("Direction shape:", direction.shape if hasattr(direction, 'shape') else len(direction))
    print("Lower first 5:", lower_st.head())
    print("Upper first 5:", upper_st.head())
    print("Direction first 5:", direction.head())
except Exception as e:
    print(f"SUPERTREND Failed: {e}")
    import traceback
    traceback.print_exc()