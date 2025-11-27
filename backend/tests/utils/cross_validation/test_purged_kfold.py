import pytest
import pandas as pd
import numpy as np
from app.services.ml.cross_validation.purged_kfold import PurgedKFold

@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    X = pd.DataFrame(np.random.rand(100, 5), index=dates, columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100), index=dates, name='label')
    # Simulate t1: label end time is 5 days after observation start
    t1 = pd.Series([d + pd.Timedelta(days=5) for d in dates], index=dates)
    return X, y, t1

def test_purged_kfold_initialization_valid(sample_data):
    X, y, t1 = sample_data
    pkf = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
    assert pkf.n_splits == 5
    assert pkf.t1.equals(t1)
    assert pkf.pct_embargo == 0.01

def test_purged_kfold_initialization_invalid_t1():
    X_df = pd.DataFrame(np.random.rand(10, 2), index=pd.to_datetime(np.arange(10)), columns=['f1', 'f2'])
    with pytest.raises(ValueError, match="t1 index must be of type DatetimeIndex."):
        PurgedKFold(n_splits=2, t1=pd.Series(np.arange(10)))


def test_purged_kfold_split_output_types(sample_data):
    X, y, t1 = sample_data
    pkf = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
    for train_idx, test_idx in pkf.split(X):
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert len(train_idx) > 0 or len(test_idx) > 0 # At least one split should have data

def test_purged_kfold_no_overlap_and_embargo(sample_data):
    X, y, t1 = sample_data
    pkf = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)

    for fold, (train_idx, test_idx) in enumerate(pkf.split(X)):
        if len(test_idx) == 0:
            continue

        # Get actual timestamps for train and test sets
        train_start_times = X.index[train_idx]
        train_end_times = t1.iloc[train_idx]
        test_start_times = X.index[test_idx]
        test_end_times = t1.iloc[test_idx]

        # 1. Purging: No training sample's (start, end) interval should overlap with any test sample's (start, end) interval
        for i in range(len(train_idx)):
            train_s = train_start_times[i]
            train_e = train_end_times.iloc[i]
            
            for j in range(len(test_idx)):
                test_s = test_start_times[j]
                test_e = test_end_times.iloc[j]

                if pd.isna(train_e) or pd.isna(test_e):
                    # Should not happen with the current fixture, but for robustness
                    continue

                # Check for overlap: (start1 <= end2 and start2 <= end1)
                assert not (train_s <= test_e and test_s <= train_e), \
                    f"Overlap detected between train[{train_s}-{train_e}] and test[{test_s}-{test_e}] in fold {fold+1}"
        
        # 2. Embargo: Training samples must not start within the embargo period after the test set
        earliest_test_start = test_start_times.min()
        latest_test_end = test_end_times.max()

        # Calculate embargo end time for this fold
        _test_duration = latest_test_end - earliest_test_start
        _embargo_seconds = pkf._get_embargo_seconds_from_duration(_test_duration, pkf.pct_embargo)
        calculated_embargo_end_time = earliest_test_start + pd.Timedelta(seconds=_embargo_seconds)

        for i in range(len(train_idx)):
            train_s = train_start_times[i]
            # Ensure no train samples start within (earliest_test_start, calculated_embargo_end_time)
            assert not (earliest_test_start <= train_s < calculated_embargo_end_time), \
                f"Train sample '{train_s}' found in embargo zone [{earliest_test_start}-{calculated_embargo_end_time}] in fold {fold+1}"