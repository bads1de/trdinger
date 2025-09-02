#!/usr/bin/env python3

import pandas as pd
import pandas_ta as ta
import numpy as np

# サンプルデータ作成
np.random.seed(42)
close = pd.Series(np.random.randn(100) + 100, index=range(100))

print("pandas-taライブラリのcmo関数をテストします")

# まずta.cmoが存在するか確認
print(f"ta.cmo存在チェック: {hasattr(ta, 'cmo')}")

try:
    # cmo	func の使用を試す
    print("\n1. ta.cmo()実行...")
    result = ta.cmo(close, length=14)
    print(f"成功! result shape: {result.shape if result is not None else 'None'}")
    print(f"result: {result[:5] if result is not None else 'None'}")

except Exception as e:
    print(f"エラー! {type(e).__name__}: {e}")

# 他のcmo関数呼び方を試す
try:
    print("\n2. ta.cmo() with period...")
    result = ta.cmo(close, period=14)
    print(f"成功! result shape: {result.shape if result is not None else 'None'}")

except Exception as e:
    print(f"エラー! {type(e).__name__}: {e}")

try:
    print("\n3. ta.cmo() with window...")
    result = ta.cmo(close, window=14)
    print(f"成功! result shape: {result.shape if result is not None else 'None'}")

except Exception as e:
    print(f"エラー! {type(e).__name__}: {e}")

# taライブラリのhelpを確認
try:
    print("\n4. cmo関数のhelp...")
    help(ta.cmo)
except Exception as e:
    print(f"ヘルプ取得エラー: {type(e).__name__}: {e}")

# taライブラリのバージョン確認
print(f"\npandas-ta version: {ta.__version__ if hasattr(ta, '__version__') else 'unknown'}")