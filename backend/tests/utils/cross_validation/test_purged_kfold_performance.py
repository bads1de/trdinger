import time
import numpy as np
import pandas as pd
import pytest
from app.services.ml.cross_validation.purged_kfold import PurgedKFold

class TestPurgedKFoldPerformance:
    def test_performance_large_dataset(self):
        """Test performance with a reasonably large dataset."""
        n_samples = 5000  # Reduced from 10000 to avoid excessive wait in CI/CD but enough to show N^2 issues
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="h")
        X = pd.DataFrame(np.random.randn(n_samples, 5), index=dates, columns=[f"feat_{i}" for i in range(5)])
        
        # Simulate labels that span 24 hours into the future
        t1 = pd.Series(dates + pd.Timedelta(hours=24), index=dates)
        
        pkf = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
        
        start_time = time.time()
        # Consume the generator to force execution
        list(pkf.split(X))
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\nPerformance test (N={n_samples}): {duration:.4f} seconds")
        
        # Current O(N^2) implementation might take > 1 second for 5000 samples depending on machine
        # We aim for significantly faster execution

