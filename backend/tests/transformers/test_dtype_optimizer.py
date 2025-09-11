import pandas as pd
import numpy as np
from backend.app.utils.data_processing.transformers.dtype_optimizer import DtypeOptimizer


def test_dtype_optimizer_optimizes_dtypes():
    """Test that DtypeOptimizer reduces memory usage by optimizing dtypes"""
    data = pd.DataFrame({
        'float_col': [1.0, 2.0, 3.0],     # Will be optimized to float32
        'int_col': [100000, 200000, 300000],  # Will be optimized to int32
        'small_int': [1, 2, 3],           # Will be optimized to int8
        'bool_col': [True, False, True]   # Will remain bool
    })

    optimizer = DtypeOptimizer()
    optimizer.fit(data)
    transformed = optimizer.transform(data)

    # Check that dtypes are optimized
    assert transformed['float_col'].dtype == 'float32'
    assert transformed['int_col'].dtype == 'int32'
    assert transformed['small_int'].dtype == 'int8'
    assert transformed['bool_col'].dtype == 'bool'


def test_dtype_optimizer_preserves_data():
    """Test that DtypeOptimizer preserves data values after optimization"""
    data = pd.DataFrame({
        'float_col': [1.5, 2.7, 3.9],
        'int_col': [100, 200, 300]
    })

    optimizer = DtypeOptimizer()
    optimizer.fit(data)
    transformed = optimizer.transform(data)

    # Check that values are preserved
    pd.testing.assert_frame_equal(transformed.astype(data.dtypes.to_dict()), data)


def test_dtype_optimizer_sklearn_compatibility():
    """Test sklearn compatibility with BaseEstimator and TransformerMixin"""
    from sklearn.base import BaseEstimator, TransformerMixin

    optimizer = DtypeOptimizer()
    assert isinstance(optimizer, BaseEstimator)
    assert isinstance(optimizer, TransformerMixin)