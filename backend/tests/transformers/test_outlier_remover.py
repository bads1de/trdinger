import pandas as pd
import numpy as np
from backend.app.utils.data_processing.transformers.outlier_remover import OutlierRemover


def test_outlier_remover_removes_outliers():
    """Test that OutlierRemover removes outliers using IQR method"""
    # Create test data with outliers
    data = pd.DataFrame({'col': [1, 2, 3, 4, 5, 100]})  # 100 is an outlier

    remover = OutlierRemover()
    remover.fit(data)
    transformed = remover.transform(data)

    # Assert that outlier is removed
    assert len(transformed) == 5
    assert not transformed['col'].isin([100]).any()


def test_outlier_remover_handles_multiple_columns():
    """Test OutlierRemover with multiple columns"""
    data = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5, 100],
        'col2': [10, 20, 30, 40, 50, 200]
    })

    remover = OutlierRemover()
    remover.fit(data)
    transformed = remover.transform(data)

    # Assert outliers are removed from both columns
    assert len(transformed) == 5
    assert not transformed['col1'].isin([100]).any()
    assert not transformed['col2'].isin([200]).any()


def test_outlier_remover_sklearn_compatibility():
    """Test sklearn compatibility with BaseEstimator and TransformerMixin"""
    from sklearn.base import BaseEstimator, TransformerMixin

    remover = OutlierRemover()
    assert isinstance(remover, BaseEstimator)
    assert isinstance(remover, TransformerMixin)