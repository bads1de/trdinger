"""
pandas-ta linear_regressionの使い方を確認
"""
import pandas_ta as ta
import pandas as pd
import numpy as np

# linear_regressionの引数を確認
help(ta.linear_regression)

# 実際に試してみる
data = pd.Series(np.random.randn(100).cumsum() + 100)
print(f"データ: {data.head()}")

try:
    # 正しい引数で試す
    result = ta.linear_regression(data, y=data, length=14, scalar=1.0)
    print(f"成功: {result}")
except Exception as e:
    print(f"エラー: {e}")

# 引数なしで試す
try:
    result = ta.linear_regression(data)
    print(f"引数なし成功: {type(result)}")
    if hasattr(result, 'columns'):
        print(f"列: {result.columns}")
        print(result.head())
except Exception as e:
    print(f"引数なしエラー: {e}")