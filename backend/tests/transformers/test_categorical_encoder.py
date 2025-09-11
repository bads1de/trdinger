import pandas as pd
from backend.app.utils.data_processing.transformers.categorical_encoder import CategoricalEncoder


def test_categorical_encoder_encodes_categories():
    """Test that CategoricalEncoder encodes categorical variables to integers"""
    data = pd.DataFrame({'cat': ['A', 'B', 'A', 'C']})

    encoder = CategoricalEncoder()
    encoder.fit(data)
    transformed = encoder.transform(data)

    # Check if categorical column is encoded to integers
    assert transformed['cat'].dtype in ['int64', 'int32', 'int8']
    assert set(transformed['cat']) == {0, 1, 2}


def test_categorical_encoder_handles_multiple_columns():
    """Test CategoricalEncoder with multiple categorical columns"""
    data = pd.DataFrame({
        'cat1': ['A', 'B', 'A'],
        'cat2': ['X', 'Y', 'X'],
        'num': [1, 2, 3]
    })

    encoder = CategoricalEncoder()
    encoder.fit(data)
    transformed = encoder.transform(data)

    # Categorical columns should be encoded, numerical should remain
    assert transformed['cat1'].dtype in ['int64', 'int32', 'int8']
    assert transformed['cat2'].dtype in ['int64', 'int32', 'int8']
    assert transformed['num'].dtype == 'int64'  # Assuming original is int64


def test_categorical_encoder_sklearn_compatibility():
    """Test sklearn compatibility with BaseEstimator and TransformerMixin"""
    from sklearn.base import BaseEstimator, TransformerMixin

    encoder = CategoricalEncoder()
    assert isinstance(encoder, BaseEstimator)
    assert isinstance(encoder, TransformerMixin)