import pandas as pd

# Test the actual parsing logic
train_period = "2023-01-01 00:00:00 ～ 2023-01-06 04:00:00"
test_period = "2023-01-06 05:00:00 ～ 2023-01-11 09:00:00"

# Parse like the test does
train_start, train_end = [pd.Timestamp(t.strip()) for t in train_period.split("～")]
test_start, test_end = [pd.Timestamp(t.strip()) for t in test_period.split("～")]

print(f"train_start: {train_start}")
print(f"train_end: {train_end}")
print(f"test_start: {test_start}")
print(f"test_end: {test_end}")
print(f"train_end <= test_start: {train_end <= test_start}")
