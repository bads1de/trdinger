from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

dates = pd.date_range("2023-01-01", periods=500, freq="h")
tss = TimeSeriesSplit(n_splits=3)

for i, (train_idx, test_idx) in enumerate(tss.split(range(500)), 1):
    print(f"Fold {i}:")
    print(f"  train indices: {train_idx[0]}-{train_idx[-1]}")
    print(f"  test indices: {test_idx[0]}-{test_idx[-1]}")
    print(f"  train period: {dates[train_idx[0]]} - {dates[train_idx[-1]]}")
    print(f"  test period: {dates[test_idx[0]]} - {dates[test_idx[-1]]}")
    print()
