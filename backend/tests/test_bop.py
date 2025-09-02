#!/usr/bin/env python3

import pandas as pd
import pandas_ta as ta
import numpy as np

# サンプルデータ作成
np.random.seed(42)
close = pd.Series(np.random.randn(100) + 100, index=range(100))
high = close + np.abs(np.random.randn(100)) * 5
low = close - np.abs(np.random.randn(100)) * 5
open_ = close + np.random.randn(100) * 2

print("pandas-taライブラリのbop関数をテストします")

# ta.bopが存在するか確認
print(f"ta.bop存在チェック: {hasattr(ta, 'bop')}")

try:
    # bop関数の使用を試す
    print("\n1. ta.bop()実行...")
    result = ta.bop(open_, high, low, close)
    print(f"成功! result shape: {result.shape if result is not None else 'None'}")
    print(f"result: {result[:5] if result is not None else 'None'}")

except Exception as e:
    print(f"エラー! {type(e).__name__}: {e}")
    print(f"詳細: {str(e)}")
    import traceback
    traceback.print_exc()

# 他の順序を試す
try:
    print("\n2. ta.bop() with kwargs...")
    result = ta.bop(open=open_, high=high, low=low, close=close)
    print(f"成功! result shape: {result.shape if result is not None else 'None'}")

except Exception as e:
    print(f"エラー! {type(e).__name__}: {e}")

# taライブラリのhelpを確認
try:
    print("\n3. bop関数のhelp...")
    help(ta.bop)
except Exception as e:
    print(f"ヘルプ取得エラー: {type(e).__name__}: {e}")

# taライブラリのバージョン確認
print(f"\npandas-ta version: {ta.__version__ if hasattr(ta, '__version__') else 'unknown'}")