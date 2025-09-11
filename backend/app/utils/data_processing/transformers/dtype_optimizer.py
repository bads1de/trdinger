import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DtypeOptimizer(BaseEstimator, TransformerMixin):
    """
    A transformer that optimizes data types to reduce memory usage.

    This transformer analyzes the data range and converts data types to more
    memory-efficient alternatives (e.g., float64 -> float32, int64 -> int32/int16/int8).
    """

    def __init__(self):
        """Initialize the DtypeOptimizer."""
        self.dtypes_ = {}

    def fit(self, X, y=None):
        """
        Fit the transformer by analyzing data ranges and determining optimal dtypes.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            Input data to fit on
        y : array-like, default=None
            Ignored. For compatibility with sklearn pipeline

        Returns:
        --------
        self : DtypeOptimizer
            Fitted transformer
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.dtypes_ = {}

        for col in X.columns:
            dtype = X[col].dtype

            # Optimize float64 to float32
            if dtype == 'float64':
                self.dtypes_[col] = 'float32'

            # Optimize int64 based on data range
            elif dtype == 'int64':
                min_val = X[col].min()
                max_val = X[col].max()

                if min_val >= -128 and max_val <= 127:
                    self.dtypes_[col] = 'int8'
                elif min_val >= -32768 and max_val <= 32767:
                    self.dtypes_[col] = 'int16'
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    self.dtypes_[col] = 'int32'
                # Keep int64 if values are too large

            # Other dtypes remain unchanged
            else:
                pass

        return self

    def transform(self, X):
        """
        Transform data types to optimized versions.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            Input data to transform

        Returns:
        --------
        pandas.DataFrame
            Data with optimized data types
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_optimized = X.copy()

        # Apply optimized dtypes
        for col in self.dtypes_:
            if col in X_optimized.columns:
                X_optimized[col] = X_optimized[col].astype(self.dtypes_[col])

        return X_optimized