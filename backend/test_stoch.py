import pandas as pd
import pandas_ta as ta
import numpy as np

# Create test data
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
close = np.random.uniform(50000, 51000, 100)
high = close * 1.01
low = close * 0.99

df = pd.DataFrame({'high': high, 'low': low, 'close': close})
print('Data shape:', df.shape)
print('Data sample:')
print(df.head())

# Test stoch
result = ta.stoch(high=df['high'], low=df['low'], close=df['close'], k=14, d=3, smooth_k=3)
print('Result type:', type(result))
print('Result shape:', result.shape if hasattr(result, 'shape') else 'No shape')
print('Result columns:', result.columns.tolist() if hasattr(result, 'columns') else 'No columns')
print('Result head:')
print(result.head())
print('Result tail:')
print(result.tail())
print('NaN count:', result.isna().sum().sum())