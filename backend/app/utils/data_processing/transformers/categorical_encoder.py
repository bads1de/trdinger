import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer that encodes categorical variables to integers using LabelEncoder.

    This transformer automatically detects categorical columns and applies label encoding
    to convert them to numerical values.
    """

    def __init__(self):
        """Initialize the CategoricalEncoder."""
        self.encoders_ = {}

    def fit(self, X, y=None):
        """
        Fit the transformer by learning the mapping for each categorical column.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            Input data to fit on
        y : array-like, default=None
            Ignored. For compatibility with sklearn pipeline

        Returns:
        --------
        self : CategoricalEncoder
            Fitted transformer
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.encoders_ = {}

        # Find categorical columns and fit encoders
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self.encoders_[col] = encoder

        return self

    def transform(self, X):
        """
        Transform categorical variables to integers.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            Input data to transform

        Returns:
        --------
        pandas.DataFrame
            Data with categorical columns encoded as integers
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_encoded = X.copy()

        # Apply encoding to categorical columns
        for col in self.encoders_:
            if col in X_encoded.columns:
                X_encoded[col] = self.encoders_[col].transform(X_encoded[col])

        return X_encoded