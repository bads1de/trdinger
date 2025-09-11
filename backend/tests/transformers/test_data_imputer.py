import pandas as pd
import numpy as np
from backend.app.utils.data_processing.transformers.data_imputer import DataImputer


def test_data_imputer_imputes_missing_values():
    """Test that DataImputer fills missing values with mean"""
    data = pd.DataFrame({'col': [1, 2, np.nan, 4]})

    imputer = DataImputer(strategy='mean')
    imputer.fit(data)
    transformed = imputer.transform(data)

    # Check that there are no missing values
    assert not transformed['col'].isnull().any()
    # Check that the imputed value is the mean of [1, 2, 4] = 2.333...
    expected_mean = (1 + 2 + 4) / 3
    assert transformed.loc[2, 'col'] == expected_mean


def test_data_imputer_handles_multiple_columns():
    """Test DataImputer with multiple columns having missing values"""
    data = pd.DataFrame({
        'col1': [1, np.nan, 3],
        'col2': [np.nan, 2, 3],
        'col3': [1, 2, 3]  # No missing values
    })

    imputer = DataImputer(strategy='mean')
    imputer.fit(data)
    transformed = imputer.transform(data)

    # All columns should have no missing values
    assert not transformed.isnull().any().any()
    # Check specific imputed values
    assert transformed.loc[1, 'col1'] == 2.0  # mean of [1, 3]
    assert transformed.loc[0, 'col2'] == 2.5  # mean of [2, 3]


def test_data_imputer_sklearn_compatibility():
    """Test sklearn compatibility with BaseEstimator and TransformerMixin"""
    from sklearn.base import BaseEstimator, TransformerMixin

    imputer = DataImputer()
    assert isinstance(imputer, BaseEstimator)
    assert isinstance(imputer, TransformerMixin)