#!/usr/bin/env python3

import pandas as pd
import pandas_ta as ta
import numpy as np

# サンプルデータ作成
np.random.seed(42)
close = pd.Series(np.random.randn(100) + 100, index=range(100))
high = close + np.abs(np.random.randn(100)) * 5
low = close - np.abs(np.random.randn(100)) * 5

print("pandas-taライブラリのultosc関数をテストします")

print(f"ta.uo存在チェック: {hasattr(ta, 'uo')}")
print(f"ta.ultosc存在チェック: {hasattr(ta, 'ultosc')}")

try:
    print("\n1. ta.uo()実行...")
    result = ta.uo(high, low, close)
    print(f"成功! result shape: {result.shape if result is not None else 'None'}")
    print(f"result: {result[:5] if result is not None else 'None'}")

except Exception as e:
    print(f"エラー! {type(e).__name__}: {e}")

try:
    print("\n2. ta.ultosc()実行...")
    result = ta.ultosc(high, low, close)
    print(f"成功! result shape: {result.shape if result is not None else 'None'}")

except Exception as e:
    print(f"エラー! {type(e).__name__}: {e}")

# helpを確認
try:
    print("\n3. uo関数のhelp...")
    help(ta.uo)
except Exception as e:
    print(f"ヘルプ取得エラー: {type(e).__name__}: {e}")

try:
    print("\n4. ultosc関数のhelp...")
    help(ta.ultosc)
except Exception as e:
    print(f"ヘルプ取得エラー: {type(e).__name__}: {e}")

print(f"\npandas-ta version: {ta.__version__ if hasattr(ta, '__version__') else 'unknown'}")