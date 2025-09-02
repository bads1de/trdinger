import pandas as pd
import numpy as np
import pandas_ta as ta

# テストデータ
np.random.seed(42)
n = 50
high = pd.Series(np.random.uniform(100, 200, n), name='high')
low = pd.Series(np.random.uniform(50, 150, n), name='low')
close = pd.Series(np.random.uniform(80, 180, n), name='close')

print("Testing ta functions with corrected parameters...")

# MOM
try:
    mom_result = ta.mom(close, length=10)
    print(f"MOM: success, range={mom_result.min():.2f} to {mom_result.max():.2f}")
except Exception as e:
    print(f"MOM error: {e}")

# ROCP
try:
    rocp_result = ta.rocp(close, length=10)
    print(f"ROCP: success, range={rocp_result.min():.2f} to {rocp_result.max():.2f}")
except Exception as e:
    print(f"ROCP error: {e}")

# ROCR
try:
    rocr_result = ta.rocr(close, length=10)
    print(f"ROCR: success, range={rocr_result.min():.2f} to {rocr_result.max():.2f}")
except Exception as e:
    print(f"ROCR error: {e}")

# ROCR100
try:
    rocr100_result = ta.rocr(close, length=10, scalar=100)
    print(f"ROCR100: success, range={rocr100_result.min():.2f} to {rocr100_result.max():.2f}")
except Exception as e:
    print(f"ROCR100 error: {e}")

# ULTOSC
try:
    ultosc_result = ta.uo(high=high, low=low, close=close, fast=7, medium=14, slow=28)
    print(f"ULTOSC: success, range={ultosc_result.min():.2f} to {ultosc_result.max():.2f}")
except Exception as e:
    print(f"ULTOSC error: {e}")

# WILLR
try:
    willr_result = ta.willr(high=high, low=low, close=close, length=14)
    print(f"WILLR: success, range={willr_result.min():.2f} to {willr_result.max():.2f}")
except Exception as e:
    print(f"WILLR error: {e}")

print("Test complete.")