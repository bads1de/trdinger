import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class DataImputer(BaseEstimator, TransformerMixin):
    """
    数値列の欠損値を補間するトランスフォーマー。

    このトランスフォーマーはsklearnのSimpleImputerを使用して、
    平均値、中央値、または最も頻繁な値で欠損値を埋めます。
    """

    def __init__(self, strategy='mean'):
        """
        DataImputerを初期化。

        Parameters:
        -----------
        strategy : str, default='mean'
            補間戦略：
            - 'mean': 欠損値を平均値で置き換え
            - 'median': 欠損値を中央値で置き換え
            - 'most_frequent': 欠損値を最も頻繁な値で置き換え
            - 'constant': 欠損値を定数値で置き換え（fill_valueを提供する必要あり）
        """
        self.strategy = strategy
        self.imputers_ = {}

    def fit(self, X, y=None):
        """
        欠損データのある各列の補間値を学習してトランスフォーマーを適合。

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            適合する入力データ
        y : array-like, default=None
            無視されます。sklearnパイプラインとの互換性のために

        Returns:
        --------
        self : DataImputer
            適合済みトランスフォーマー
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.imputers_ = {}

        # 欠損値のある各列のインピューターを適合
        for col in X.columns:
            if X[col].isnull().any():
                imputer = SimpleImputer(strategy=self.strategy)
                imputer.fit(X[[col]])
                self.imputers_[col] = imputer

        return self

    def transform(self, X):
        """
        データの欠損値を補間。

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            変換する入力データ

        Returns:
        --------
        pandas.DataFrame
            欠損値が補間されたデータ
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_imputed = X.copy()

        # 適合時に欠損値があった列に補間を適用
        for col in self.imputers_:
            if col in X_imputed.columns:
                # SimpleImputerは2D配列を返すので、1Dにravelする必要がある
                X_imputed[col] = self.imputers_[col].transform(X_imputed[[col]]).ravel()

        return X_imputed