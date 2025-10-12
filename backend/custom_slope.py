"""
slope計算の代替方法を探索
"""
import pandas as pd
import numpy as np

# 単純な線形回帰スロープの実装
def calculate_slope(data, length=14):
    """指定された期間の線形回帰スロープを計算"""
    if len(data) < length:
        return pd.Series([np.nan] * len(data), index=data.index)

    slopes = [np.nan] * (length - 1)

    for i in range(length - 1, len(data)):
        window = data[i-length+1:i+1]
        x = np.arange(length)
        slope = np.polyfit(x, window, 1)[0]  # 1次多項式の係数（スロープ）
        slopes.append(slope)

    return pd.Series(slopes, index=data.index)

# テスト
data = pd.Series(np.random.randn(100).cumsum() + 100)
result = calculate_slope(data, 14)
print(f"スロープ計算: {result.tail()}")