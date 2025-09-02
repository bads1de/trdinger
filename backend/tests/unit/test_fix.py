import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np

# Directly import the direction
import importlib.util
spec = importlib.util.spec_from_file_location("trend", "app/services/indicators/technical_indicators/trend.py")
trend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trend_module)
TrendIndicators = trend_module.TrendIndicators

# Test T3 parameter validation
print("Testing T3 parameter validation...")

# Invalid length
try:
    TrendIndicators.t3(pd.Series([1, 2, 3]), length=0)
    print("ERROR: Should have raised ValueError for length=0")
except ValueError as e:
    print(f"GOOD: Raised ValueError for length=0: {e}")

# Invalid a
try:
    TrendIndicators.t3(pd.Series([1, 2, 3]), a=-0.1)
    print("ERROR: Should have raised ValueError for a=-0.1")
except ValueError as e:
    print(f"GOOD: Raised ValueError for a=-0.1: {e}")

try:
    TrendIndicators.t3(pd.Series([1, 2, 3]), a=1.5)
    print("ERROR: Should have raised ValueError for a=1.5")
except ValueError as e:
    print(f"GOOD: Raised ValueError for a=1.5: {e}")

# Valid parameters
try:
    result = TrendIndicators.t3(pd.Series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190]), length=5, a=0.7)
    print(f"Good: T3 calculation successful: {len(result)} points")
except Exception as e:
    print(f"ERROR in T3 calculation: {e}")

print("\nTesting TLB parameter validation...")

# Invalid length
try:
    TrendIndicators.tlb(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), length=0)
    print("ERROR: Should have raised ValueError for length=0")
except ValueError as e:
    print(f"GOOD: Raised ValueError for length=0: {e}")

print("\nTesting ZLMA...")
try:
    result = TrendIndicators.zlma(pd.Series([100, 110, 120, 130, 140, 150]))
    print(f"Good: ZLMA calculation successful: {len(result)} points")
except Exception as e:
    print(f"ERROR in ZLMA calculation: {e}")

print("All tests completed.")