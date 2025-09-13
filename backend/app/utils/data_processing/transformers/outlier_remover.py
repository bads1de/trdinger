import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    IQRメソッドを使用して数値列から外れ値を除去するトランスフォーマー。

    このトランスフォーマーは四分位範囲（IQR）に基づいて外れ値を識別し、
    データセットから外れ値を含む行を除去します。
    """

    def __init__(self, factor=1.5):
        """
        OutlierRemoverを初期化。

        Parameters:
        -----------
        factor : float, default=1.5
            外れ値境界を決定するためにIQRに掛ける係数。
            下限 = Q1 - factor * IQR
            上限 = Q3 + factor * IQR
        """
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        """
        各数値列の外れ値境界を計算してトランスフォーマーを適合。

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            適合する入力データ
        y : array-like, default=None
            無視されます。sklearnパイプラインとの互換性のために

        Returns:
        --------
        self : OutlierRemover
            適合済みトランスフォーマー
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.bounds_ = {}

        # 数値列のみの境界を計算
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
        適合された境界に基づいてデータから外れ値を除去。

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            変換する入力データ

        Returns:
        --------
        pandas.DataFrame
            外れ値が除去されたデータ
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_clean = X.copy()

        # いずれかの数値列に外れ値がある行をフィルタリング
        for col in self.bounds_:
            if col in X_clean.columns:
                lower, upper = self.bounds_[col]
                X_clean = X_clean[(X_clean[col] >= lower) & (X_clean[col] <= upper)]

        return X_clean