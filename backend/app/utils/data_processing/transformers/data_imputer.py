import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class DataImputer(BaseEstimator, TransformerMixin):
    """
    A transformer that imputes missing values in numerical columns.

    This transformer uses sklearn's SimpleImputer to fill missing values
    with mean, median, or most frequent values.
    """

    def __init__(self, strategy='mean'):
        """
        Initialize the DataImputer.

        Parameters:
        -----------
        strategy : str, default='mean'
            The imputation strategy:
            - 'mean': replace missing values with mean
            - 'median': replace missing values with median
            - 'most_frequent': replace missing values with most frequent value
            - 'constant': replace missing values with a constant value (fill_value must be provided)
        """
        self.strategy = strategy
        self.imputers_ = {}

    def fit(self, X, y=None):
        """
        Fit the transformer by learning imputation values for each column with missing data.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            Input data to fit on
        y : array-like, default=None
            Ignored. For compatibility with sklearn pipeline

        Returns:
        --------
        self : DataImputer
            Fitted transformer
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.imputers_ = {}

        # Fit imputer for each column that has missing values
        for col in X.columns:
            if X[col].isnull().any():
                imputer = SimpleImputer(strategy=self.strategy)
                imputer.fit(X[[col]])
                self.imputers_[col] = imputer

        return self

    def transform(self, X):
        """
        Impute missing values in the data.

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            Input data to transform

        Returns:
        --------
        pandas.DataFrame
            Data with missing values imputed
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_imputed = X.copy()

        # Apply imputation to columns that had missing values during fit
        for col in self.imputers_:
            if col in X_imputed.columns:
                # SimpleImputer returns 2D array, so we need to ravel it to 1D
                X_imputed[col] = self.imputers_[col].transform(X_imputed[[col]]).ravel()

        return X_imputed