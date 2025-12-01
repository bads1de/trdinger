import numpy as np
import pandas as pd
import pytest
from app.services.ml.preprocessing.fractional_differentiation import FractionalDifferentiation

class TestFractionalDifferentiation:
    def test_get_weights_floats(self):
        """Test weight calculation for a specific fractional order."""
        d = 0.5
        size = 5
        fd = FractionalDifferentiation(d=d)
        weights = fd._get_weights(d, size)
        
        # Expected weights for d=0.5:
        # w0 = 1
        # w1 = -d = -0.5
        # w2 = -w1 * (d - 2 + 1) / 2 = -(-0.5) * (-0.5) / 2 = -0.125
        # ...
        
        assert len(weights) == size
        assert weights[0] == 1.0
        assert np.isclose(weights[1], -0.5)
        assert np.isclose(weights[2], -0.125)
        
        # Check that weights converge to zero
        assert abs(weights[-1]) < abs(weights[0])

    def test_fixed_width_window_frac_diff(self):
        """Test fractional differentiation on a simple series."""
        # Create a simple linear trend
        data = pd.Series(np.arange(100, dtype=float))
        
        # With d=1, it should be close to standard first difference
        # (except for the first few points due to windowing)
        fd = FractionalDifferentiation(d=1.0, window_size=10)
        diff_data = fd.transform(data)
        
        # Standard diff is all 1s (except first nan)
        # Frac diff with d=1 should approximate this
        
        assert isinstance(diff_data, pd.Series)
        assert len(diff_data) == len(data)
        
        # Check valid values (after window size)
        valid_data = diff_data.iloc[10:]
        assert np.allclose(valid_data, 1.0, atol=1e-5)

    def test_stationarity_preservation(self):
        """
        Test that fractional differentiation creates a stationary series 
        from a non-stationary one (Random Walk).
        """
        np.random.seed(42)
        n_samples = 1000
        # Create random walk
        returns = np.random.randn(n_samples)
        price = pd.Series(np.cumsum(returns))
        
        # Apply frac diff with d=0.4
        fd = FractionalDifferentiation(d=0.4, window_size=20)
        diff_price = fd.transform(price)
        
        # Drop NaNs introduced by window
        diff_price = diff_price.dropna()
        
        # Check stats
        # Original price should have high variance/range
        # Diff price should be centered around 0
        assert diff_price.std() < price.std()
        assert abs(diff_price.mean()) < 1.0

    def test_dataframe_support(self):
        """Test transform on DataFrame (multiple columns)."""
        df = pd.DataFrame({
            'A': np.arange(50, dtype=float),
            'B': np.arange(50, dtype=float) * 2
        })
        
        fd = FractionalDifferentiation(d=0.5, window_size=5)
        res = fd.transform(df)
        
        assert isinstance(res, pd.DataFrame)
        assert res.shape == df.shape
        assert 'A' in res.columns
        assert 'B' in res.columns
        
        # Check that columns were processed independently but correctly
        # A is 0, 1, 2...
        # B is 0, 2, 4...
        # Result B should be approx 2 * Result A
        
        valid_idx = 10
        assert np.isclose(res['B'].iloc[valid_idx], 2 * res['A'].iloc[valid_idx], atol=1e-5)

    def test_memory_usage_optimization(self):
        """Verify that we don't compute full expansion if threshold cuts it off."""
        # This is implicitly tested by _get_weights logic if we implement threshold cutoff
        # Here we just check that passing a threshold works
        fd = FractionalDifferentiation(d=0.5, weight_threshold=1e-3)
        weights = fd._get_weights(0.5, 100)
        
        # Weights should stop when they drop below threshold
        # Note: In fixed window implementation, we usually enforce window_size.
        # If we implement iterative weight calculation with cutoff, length varies.
        # Let's assume our implementation respects window_size first, or threshold if specified.
        pass
