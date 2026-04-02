from typing import Dict, Mapping, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _infer_integer_dtype(
    col_min: int,
    col_max: int,
    prefer_unsigned_integers: bool,
) -> Optional[str]:
    """整数列に対して、より省メモリな dtype 名を推定する。"""
    if prefer_unsigned_integers and col_min >= 0:
        if col_max < 255:
            return "uint8"
        if col_max < 65535:
            return "uint16"
        if col_max < 4294967295:
            return "uint32"

    if col_min > -128 and col_max < 127:
        return "int8"
    if col_min > -32768 and col_max < 32767:
        return "int16"
    if col_min > -2147483648 and col_max < 2147483647:
        return "int32"
    return None


def _get_column_series(df: pd.DataFrame, col: str) -> pd.Series:
    """重複列を含む場合は先頭列を使って dtype を判定する。"""
    col_data = df[col]
    if isinstance(col_data, pd.DataFrame):
        return col_data.iloc[:, 0]  # type: ignore
    return col_data


def build_optimized_dtype_map(
    df: pd.DataFrame,
    *,
    prefer_unsigned_integers: bool = False,
    optimize_all_numeric: bool = False,
) -> Dict[str, str]:
    """DataFrame から最適化後の dtype マップを構築する。

    optimize_all_numeric=False は既存の DtypeOptimizer と同等の挙動、
    True は preprocessing pipeline 側のより積極的な最適化に合わせる。
    """
    dtypes: Dict[str, str] = {}

    for col in df.columns:
        col_series = _get_column_series(df, col)

        if col_series.dtype == "float64" or (
            optimize_all_numeric and pd.api.types.is_float_dtype(col_series)
        ):
            optimized = pd.to_numeric(col_series, downcast="float")
            if optimized.dtype != col_series.dtype:  # pyright: ignore[reportAttributeAccessIssue]
                dtypes[col] = str(optimized.dtype)  # pyright: ignore[reportAttributeAccessIssue]
            continue

        if col_series.dtype == "int64" or (
            optimize_all_numeric and pd.api.types.is_integer_dtype(col_series)
        ):
            col_min = int(col_series.min())
            col_max = int(col_series.max())
            optimized_dtype = _infer_integer_dtype(
                col_min,
                col_max,
                prefer_unsigned_integers=prefer_unsigned_integers,
            )
            if optimized_dtype is not None:
                dtypes[col] = optimized_dtype

    return dtypes


def apply_optimized_dtypes(
    df: pd.DataFrame,
    dtype_map: Mapping[str, str],
) -> pd.DataFrame:
    """dtype マップを DataFrame に適用する。"""
    valid_dtypes = {k: v for k, v in dtype_map.items() if k in df.columns}
    return df.astype(valid_dtypes)


def optimize_dataframe_dtypes(
    df: pd.DataFrame,
    *,
    prefer_unsigned_integers: bool = False,
    optimize_all_numeric: bool = False,
) -> pd.DataFrame:
    """DataFrame をその場で最適化した dtype に変換する。"""
    dtype_map = build_optimized_dtype_map(
        df,
        prefer_unsigned_integers=prefer_unsigned_integers,
        optimize_all_numeric=optimize_all_numeric,
    )
    return apply_optimized_dtypes(df, dtype_map)


class DtypeOptimizer(BaseEstimator, TransformerMixin):
    # 後方互換性のための別名
    DataTypeOptimizer = None
    """
    メモリ使用量を削減するためにデータ型を最適化するトランスフォーマー。

    このトランスフォーマーはデータの範囲を分析し、よりメモリ効率の高い代替データ型に
    変換します（例: float64 -> float32, int64 -> int32/int16/int8）。
    """

    def __init__(self):
        """DtypeOptimizerを初期化。"""

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

        self.dtypes_ = build_optimized_dtype_map(X)
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

        return apply_optimized_dtypes(X, self.dtypes_)



