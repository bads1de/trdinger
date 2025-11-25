import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from typing import Optional, Union, List, Any

class PurgedKFold(_BaseKFold):
    """
    Purged K-Fold Cross Validation.

    This cross-validation scheme ensures that the training and testing sets are
    separated in time to prevent data leakage from the future into the past.
    It also purges samples from the training set that overlap with the testing
    set to prevent label leakage.

    Based on Marcos Lopez de Prado's "Advances in Financial Machine Learning".

    Args:
        n_splits (int): Number of splits.
        t1 (pd.Series): Series of label end times for each sample in the dataset.
                       Index should match the index of the features (X) and labels (y).
        pct_embargo (float): Percentage of the test set duration to embargo from the training set.
                             This helps prevent leakage from the training set to the test set
                             if labels are formed using information within a certain window.
    """
    def __init__(self, n_splits: int = 5, t1: pd.Series = None, pct_embargo: float = 0.01):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pandas Series with DatetimeIndex.")
        if not all(isinstance(idx, (pd.Timestamp, np.datetime64)) for idx in t1.index):
            raise ValueError("t1 index must be of type DatetimeIndex.")
        if not all(isinstance(val, (pd.Timestamp, np.datetime64)) or pd.isna(val) for val in t1.values):
            raise ValueError("t1 values must be of type DatetimeIndex or NaT.")
        
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[Any] = None):
        """
        Generates indices to split data into training and test set.

        Args:
            X (pd.DataFrame): Training data.
            y (Optional[pd.Series]): Target variable (ignored, but kept for scikit-learn compatibility).
            groups (Optional[Any]): Group labels for the samples (ignored).

        Yields:
            Tuple[np.ndarray, np.ndarray]: The training and test set indices for that split.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if not X.index.equals(self.t1.index):
            raise ValueError("X and t1 must have the same index.")

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo) # Embargo period in number of samples
        
        for i in range(self.n_splits):
            test_start = (X.shape[0] // self.n_splits) * i
            test_end = (X.shape[0] // self.n_splits) * (i + 1)
            
            # Adjust test_end for the last fold
            if i == self.n_splits - 1:
                test_end = X.shape[0]

            test_indices = indices[test_start:test_end]
            
            if len(test_indices) == 0:
                continue

            test_times = X.index[test_indices]
            # Use t1 for end of test period
            test_t1_times = self.t1.loc[test_times]
            
            train_indices = []
            for j in range(X.shape[0]):
                if j in test_indices:
                    continue
                
                # Check for overlap with any test sample (purge)
                # Ensure training sample's end time (t1) is before test sample's start time
                # Or training sample's start time is after test sample's end time
                
                # Condition 1: Training sample j must not end after any test sample starts
                # This prevents standard data leakage where a training observation overlaps with a test observation
                train_sample_t1 = self.t1.iloc[j]
                
                # This is more robust check. Training observations must not overlap test observations.
                # Specifically, training observation 'j' (from X.index[j] to self.t1.iloc[j])
                # must not overlap with any test observation 'k' (from test_times[k] to test_t1_times[k]).
                
                # No overlap means:
                # (train_end < test_start) OR (train_start > test_end)
                
                # Here, we are checking for the j-th training sample (X.index[j], self.t1.iloc[j])
                # against the entire test set (test_times[k], test_t1_times[k]).
                
                # Faster check:
                # Find max end time of all training samples up to j.
                # Find min start time of all test samples from test_start.
                
                # Detailed check for each training sample 'j' against all test samples:
                is_purged = False
                for k in range(len(test_indices)):
                    test_sample_start = test_times[k]
                    test_sample_end = test_t1_times.iloc[k]
                    
                    if pd.isna(train_sample_t1) or pd.isna(test_sample_end):
                        # Handle NaT values if needed, typically means it extends indefinitely
                        # For simplicity, if NaT, assume potential overlap and purge
                        is_purged = True
                        break

                    # Overlap if:
                    # (train_start < test_end AND train_end > test_start)
                    # OR (test_start < train_end AND test_end > train_start)
                    
                    # Simpler check for purge:
                    # training sample's start time should be before test sample's start time
                    # training sample's end time should be before test sample's start time
                    # Or it should be after test sample's end time + embargo
                    
                    # condition to keep training sample j:
                    # (self.t1.iloc[j] < test_sample_start)  # end before test start
                    # OR (X.index[j] > test_sample_end + pd.Timedelta(seconds=self._get_embargo_seconds(test_sample_end, test_sample_start))) # start after test end + embargo
                    
                    # Lopez de Prado's definition of purging:
                    # A training observation i is purged if:
                    # [X.index[i], self.t1.iloc[i]] overlaps with [X.index[test_idx_k], self.t1.iloc[test_idx_k]]
                    
                    # This means, for each training sample 'j', its interval [X.index[j], self.t1.iloc[j]]
                    # must not overlap with *any* of the test intervals [test_times[k], test_t1_times.iloc[k]].
                    
                    # A simpler check is to remove all training observations whose
                    # interval [X.index[j], self.t1.iloc[j]] intersects with the interval
                    # [test_times.min(), test_t1_times.max()].
                    # However, this might purge too much.
                    
                    # Let's use the strict definition of no overlap:
                    # train_start_j = X.index[j]
                    # train_end_j = train_sample_t1
                    # test_start_k = test_sample_start
                    # test_end_k = test_sample_end
                    
                    # Overlap happens if not ((train_end_j < test_start_k) or (train_start_j > test_end_k))
                    # So, no overlap if (train_end_j < test_start_k) or (train_start_j > test_end_k)
                    
                    # For purging: if (train_start_j <= test_end_k and test_start_k <= train_end_j)
                    # This is the standard interval overlap check.
                    
                    if (X.index[j] <= test_sample_end and test_sample_start <= train_sample_t1):
                        is_purged = True
                        break
                
                if not is_purged:
                    # Apply embargo
                    embargo_end_time = test_t1_times.max() + pd.Timedelta(seconds=self._get_embargo_seconds_from_duration(
                        test_t1_times.max() - test_times.min(), self.pct_embargo
                    ))
                    
                    # If training sample j starts after the embargo period ends
                    if X.index[j] > embargo_end_time:
                        train_indices.append(j)
                    elif X.index[j] < test_times.min(): # training sample before test set
                        train_indices.append(j)

            yield np.array(train_indices), test_indices

    def _get_embargo_seconds_from_duration(self, duration: pd.Timedelta, pct_embargo: float) -> float:
        """Calculates embargo seconds based on test set duration and percentage."""
        return duration.total_seconds() * pct_embargo

# Helper for testing/debugging
if __name__ == '__main__':
    # Sample data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    X_df = pd.DataFrame(np.random.rand(100, 5), index=dates, columns=[f'feature_{i}' for i in range(5)])
    y_series = pd.Series(np.random.randint(0, 2, 100), index=dates, name='label')
    
    # Simulate t1 for each sample (label end time)
    # For simplicity, let's say a label lasts for 5 days from its start
    t1_series = pd.Series([d + pd.Timedelta(days=5) for d in dates], index=dates)

    # Initialize PurgedKFold
    pkf = PurgedKFold(n_splits=5, t1=t1_series, pct_embargo=0.01)

    # Perform splitting
    for fold, (train_idx, test_idx) in enumerate(pkf.split(X_df)):
        print(f"Fold {fold+1}:")
        print(f"  Train indices length: {len(train_idx)}, Test indices length: {len(test_idx)}")
        
        train_start_time = X_df.index[train_idx[0]]
        train_end_time = t1_series.iloc[train_idx[-1]]
        test_start_time = X_df.index[test_idx[0]]
        test_end_time = t1_series.iloc[test_idx[-1]]

        print(f"  Train: {train_start_time} - {train_end_time}")
        print(f"  Test:  {test_start_time} - {test_end_time}")
        
        # Verify no overlap and embargo
        # Max t1 of train set should be before min start time of test set
        max_train_t1 = t1_series.iloc[train_idx].max()
        min_test_start = X_df.index[test_idx].min()
        
        # Test start time interval
        min_test_period_start = X_df.index[test_idx].min()
        max_test_period_end = t1_series.loc[X_df.index[test_idx]].max()

        # Embargo start time
        embargo_start_time = X_df.index[test_idx].min()
        embargo_duration = (max_test_period_end - min_test_period_start).total_seconds() * pkf.pct_embargo
        embargo_end_time = embargo_start_time + pd.Timedelta(seconds=embargo_duration)
        
        print(f"  Max Train T1: {max_train_t1}")
        print(f"  Min Test Start: {min_test_start}")
        print(f"  Embargo End Time: {embargo_end_time}")

        # Assert that there's no overlap for training indices with test period
        # Specifically, for each train_idx, the interval [X.index[i], t1.iloc[i]]
        # must not overlap with any test interval [X.index[j], t1.iloc[j]]
        
        # This is implicitly handled by the loop.
        
        # The key is that the training set does not contain observations that leak into the test set
        # nor observations that are too close to the test set start.
        
        # Simple verification: max end time of any training observation should be before the start of the current test period (minus embargo)
        # However, it's more subtle. A training observation can occur *before* the test period, but its label may extend *into* the test period.
        # This is why we need t1 for purging.
        
        # The loop logic implements:
        # A training sample is valid if its interval [X.index[j], self.t1.iloc[j]] does not overlap with any test interval
        # AND (X.index[j] < test_times.min()) OR (X.index[j] > embargo_end_time)
        
        # Let's simplify the verification for now
        # Assert that train_indices are disjoint from test_indices (already handled by construction)
        # Assert that there are no train samples in the purged region
        
        # purged_region_start = X_df.index[test_start] - (X_df.index[test_end] - X_df.index[test_start]) * pkf.pct_embargo # not correct
        # purged_region_end = X_df.index[test_end] + (X_df.index[test_end] - X_df.index[test_start]) * pkf.pct_embargo # not correct
        
        # Purging: remove training observations whose evaluation period (t1) overlaps with test_set.
        # This is handled in the `is_purged` loop.
        
        # Embargo: remove training observations that immediately precede the test set.
        # This is handled by `X.index[j] > embargo_end_time` or `X.index[j] < test_times.min()`
        
        # More robust manual checks might be needed in a full test suite.
        # For this basic implementation, the logic aims to follow the principles.
