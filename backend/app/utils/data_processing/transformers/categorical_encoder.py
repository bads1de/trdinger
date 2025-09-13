import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    LabelEncoderを使用してカテゴリ変数を整数にエンコードするトランスフォーマー。

    このトランスフォーマーは自動的にカテゴリ列を検出し、ラベルエンコーディングを適用して
    数値に変換します。
    """

    def __init__(self):
        """CategoricalEncoderを初期化。"""
        self.encoders_ = {}

    def fit(self, X, y=None):
        """
        各カテゴリ列のマッピングを学習してトランスフォーマーを適合。

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            適合する入力データ
        y : array-like, default=None
            無視されます。sklearnパイプラインとの互換性のために

        Returns:
        --------
        self : CategoricalEncoder
            適合済みトランスフォーマー
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.encoders_ = {}

        # カテゴリ列を検出し、エンコーダーを適合
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self.encoders_[col] = encoder

        return self

    def transform(self, X):
        """
        カテゴリ変数を整数に変換。

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            変換する入力データ

        Returns:
        --------
        pandas.DataFrame
            カテゴリ列が整数としてエンコードされたデータ
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_encoded = X.copy()

        # カテゴリ列にエンコーディングを適用
        for col in self.encoders_:
            if col in X_encoded.columns:
                X_encoded[col] = self.encoders_[col].transform(X_encoded[col])

        return X_encoded