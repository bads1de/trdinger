import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from app.services.indicators.technical_indicators.trend import TrendIndicators

# サンプルデータ作成
np.random.seed(42)  # 再現性のために
n = 100
high = pd.Series(np.random.uniform(100, 110, n))
low = pd.Series(np.random.uniform(95, 105, n))
open_vals = pd.Series(np.random.uniform(97, 103, n))
close = pd.Series(np.random.uniform(98, 102, n))
volume = pd.Series(np.random.uniform(1000, 10000, n))
data = close.copy()

indicators = [
    ('ohlc4', lambda: TrendIndicators.ohlc4(open_vals, high, low, close)),
    ('pwma', lambda: TrendIndicators.pwma(data)),
    ('range_func', lambda: TrendIndicators.range_func(high)),
    ('ssf', lambda: TrendIndicators.ssf(data)),
    ('tlb', lambda: TrendIndicators.tlb(high, low, close)),
    ('vidya', lambda: TrendIndicators.vidya(data)),
    ('vwma', lambda: TrendIndicators.vwma(data, volume)),
    ('wcp', lambda: TrendIndicators.wcp(data)),
]

print("Testing TrendIndicators fixes...")
for name, func in indicators:
    try:
        result = func()
        success = True
        length = len(result)
        all_nan = result.isna().all()
        if not result.isna().all():
            min_val = result.min()
            max_val = result.max()
            print(".2f")
        else:
            print(f"{name}: success, length={length}, all_nan={all_nan}")
    except Exception as e:
        print(f"{name}: ERROR - {e}")
        continue

print("All tests completed.")