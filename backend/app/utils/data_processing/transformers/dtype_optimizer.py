import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DtypeOptimizer(BaseEstimator, TransformerMixin):
    """
    メモリ使用量を削減するためにデータ型を最適化するトランスフォーマー。

    このトランスフォーマーはデータの範囲を分析し、よりメモリ効率の高い代替データ型に
    変換します（例: float64 -> float32, int64 -> int32/int16/int8）。
    """

    def __init__(self):
        """DtypeOptimizerを初期化。"""
        self.dtypes_ = {}

    def fit(self, X, y=None):
        """
        データの範囲を分析して最適なdtypeを決定してトランスフォーマーを適合。

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            適合する入力データ
        y : array-like, default=None
            無視されます。sklearnパイプラインとの互換性のために

        Returns:
        --------
        self : DtypeOptimizer
            適合済みトランスフォーマー
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.dtypes_ = {}

        for col in X.columns:
            dtype = X[col].dtype

            # float64をfloat32に最適化
            if dtype == "float64":
                self.dtypes_[col] = "float32"

            # データ範囲に基づいてint64を最適化
            elif dtype == "int64":
                min_val = X[col].min()
                max_val = X[col].max()

                if min_val >= -128 and max_val <= 127:
                    self.dtypes_[col] = "int8"
                elif min_val >= -32768 and max_val <= 32767:
                    self.dtypes_[col] = "int16"
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    self.dtypes_[col] = "int32"
                # 値が大きすぎる場合はint64を保持

            # その他のdtypeは変更なし
            else:
                pass

        return self

    def transform(self, X):
        """
        データ型を最適化されたバージョンに変換。

        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            変換する入力データ

        Returns:
        --------
        pandas.DataFrame
            最適化されたデータ型のデータ
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_optimized = X.copy()

        # 最適化されたdtypeを適用
        for col in self.dtypes_:
            if col in X_optimized.columns:
                X_optimized[col] = X_optimized[col].astype(self.dtypes_[col])

        return X_optimized
