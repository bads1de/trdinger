#!/usr/bin/env python3
import pandas_ta as ta
import pandas as pd
import numpy as np

print("Testing pandas-ta CHOP and VORTEX functions")

# Create sample data
np.random.seed(42)
high = pd.Series([101, 102, 103, 104, 105])
low = pd.Series([99, 98, 97, 96, 95])
close = pd.Series([100, 101, 102, 103, 104])

print("Sample data:")
print("High:", high.values)
print("Low:", low.values)
print("Close:", close.values)

# Test CHOP
try:
    result = ta.chop(high, low, close, length=14)
    print("CHOP call successful, result:", result.values if hasattr(result, 'values') else result)
except Exception as e:
    print("CHOP call failed:", str(e))
    print("Error type:", type(e).__name__)

# Test VORTEX
try:
    result = ta.vortex(high, low, close, length=14)
    print("VORTEX call successful, result:", result.values if hasattr(result, 'values') else result)
except Exception as e:
    print("VORTEX call failed:", str(e))
    print("Error type:", type(e).__name__)

print("Test completed")