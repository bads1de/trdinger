import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    A transformer that removes outliers from numerical columns using IQR method.

    This transformer identifies outliers based on the Interquartile Range (IQR) and
    removes rows containing outliers from the dataset.
    """

    def __init__(self, factor=1.5):
        """
        Initialize the OutlierRemover.

        Parameters:
        -----------
        factor : float, default=1.5
            The factor to multiply IQR by to determine outlier bounds.
            Lower bound = Q1 - factor * IQR
            Upper bound = Q3 + factor * IQR
        """
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        """
        Fit the transformer by calculating outlier bounds for each numerical column.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            Input data to fit on
        y : array-like, default=None
            Ignored. For compatibility with sklearn pipeline

        Returns:
        --------
        self : OutlierRemover
            Fitted transformer
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.bounds_ = {}

        # Calculate bounds for numerical columns only
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR

            self.bounds_[col] = (lower_bound, upper_bound)

        return self

    def transform(self, X):
        """
        Remove outliers from the data based on fitted bounds.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            Input data to transform

        Returns:
        --------
        pandas.DataFrame
            Data with outliers removed
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_clean = X.copy()

        # Filter out rows with outliers in any numerical column
        for col in self.bounds_:
            if col in X_clean.columns:
                lower, upper = self.bounds_[col]
                X_clean = X_clean[(X_clean[col] >= lower) & (X_clean[col] <= upper)]

        return X_clean