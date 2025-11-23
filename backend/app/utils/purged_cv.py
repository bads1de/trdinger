import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from typing import Optional, Union, Generator, Tuple

class PurgedKFold(_BaseKFold):
    """
    Purged K-Fold Cross Validation for financial time series data.
    Prevents leakage from test set to training set and between training folds
    by purging samples within a certain time window around each test fold.

    Args:
        n_splits (int): Number of folds.
        t1 (pd.Series): A Series of timestamps marking the end of the labels.
                        Index must align with X and y.
        embargo_pct (float): Percentage of the test set duration to embargo from the training set.
                             E.g., 0.01 means 1% of the test period duration will be embargoed.
    """
    def __init__(self, n_splits: int = 3, t1: Optional[pd.Series] = None, embargo_pct: float = 0.0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        if t1 is None:
            raise ValueError("t1 (end times of labels) must be provided for PurgedKFold.")
        self.t1 = pd.Series(t1) # Ensure t1 is a Series
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex for PurgedKFold.")
        if not X.index.equals(self.t1.index):
            raise ValueError("Index of X and t1 must be aligned.")

        indices = np.arange(X.shape[0])
        n_samples = X.shape[0]
        
        # Calculate fold sizes for K-Fold
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx_raw = np.concatenate([indices[:start], indices[stop:]])
            
            # --- Purging and Embargo Logic ---
            # Times for the current test set
            test_start_t0 = X.index[test_idx[0]]  # Start time of the first observation in test set
            test_end_t1 = self.t1.iloc[test_idx[-1]] # End time of the last label in test set

            # Calculate embargo period duration
            # Embargo duration is a percentage of the *test set's observation span*
            test_observation_span = X.index[test_idx[-1]] - X.index[test_idx[0]]
            embargo_duration = test_observation_span * self.embargo_pct
            
            # The earliest time a training observation can start after the embargo period
            min_train_t0_after_embargo = test_end_t1 + embargo_duration

            train_idx_purged_embargoed = []
            for j in train_idx_raw:
                train_t0_j = X.index[j]  # Start time of current training observation
                train_t1_j = self.t1.iloc[j] # End time of current training label

                # Purging Condition:
                # Training label [train_t0_j, train_t1_j] must NOT overlap with test label [test_start_t0, test_end_t1]
                # Overlap exists if (train_t0_j <= test_end_t1) AND (train_t1_j >= test_start_t0)
                # So, if this condition is TRUE, we skip (purge) the sample.
                is_purged = (train_t0_j <= test_end_t1) and (train_t1_j >= test_start_t0)
                if is_purged:
                    continue

                # Embargo Condition:
                # Training observation start time (train_t0_j) must NOT be within the embargo period
                # The embargo period is [test_end_t1, min_train_t0_after_embargo]
                is_embargoed = (train_t0_j >= test_end_t1) and (train_t0_j <= min_train_t0_after_embargo)
                if is_embargoed:
                    continue
                
                train_idx_purged_embargoed.append(j)

            yield np.array(train_idx_purged_embargoed, dtype=int), test_idx
            
            current += fold_size

    def get_n_splits(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None) -> int:
        return self.n_splits