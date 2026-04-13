from typing import Dict, Mapping, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _infer_integer_dtype(
    col_min: int,
    col_max: int,
    prefer_unsigned_integers: bool,
) -> Optional[str]:
    """
    整数列に対して、より省メモリな dtype 名を推定する

    カラムの最小値と最大値に基づいて、適切な整数型を推定します。

    Args:
        col_min: カラムの最小値
        col_max: カラムの最大値
        prefer_unsigned_integers: 符号なし整数を優先するかどうか

    Returns:
        Optional[str]: 最適なdtype名、該当しない場合はNone

    推定ルール:
        - prefer_unsigned_integers=Trueかつcol_min>=0の場合:
            - col_max < 255: uint8
            - col_max < 65535: uint16
            - col_max < 4294967295: uint32
        - その他の場合:
            - -128 < col_min < 127: int8
            - -32768 < col_min < 32767: int16
            - -2147483648 < col_min < 2147483647: int32
    """
    if prefer_unsigned_integers and col_min >= 0:
        if col_max <= 255:
            return "uint8"
        if col_max <= 65535:
            return "uint16"
        if col_max <= 4294967295:
            return "uint32"

    if col_min >= -128 and col_max <= 127:
        return "int8"
    if col_min >= -32768 and col_max <= 32767:
        return "int16"
    if col_min >= -2147483648 and col_max <= 2147483647:
        return "int32"
    return None


def _get_column_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    重複列を含む場合は先頭列を使って dtype を判定する

    カラムがDataFrameの場合は先頭列を返し、Seriesの場合はそのまま返します。

    Args:
        df: DataFrame
        col: カラム名

    Returns:
        pd.Series: カラムデータ（Series）
    """
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
    """
    DataFrame から最適化後の dtype マップを構築する

    DataFrameの各カラムを分析して、メモリ効率の良いdtypeを推定します。

    Args:
        df: DataFrame
        prefer_unsigned_integers: 符号なし整数を優先するかどうか（デフォルト: False）
        optimize_all_numeric: 全ての数値型を最適化するかどうか（デフォルト: False）

    Returns:
        Dict[str, str]: カラム名からdtype名へのマッピング

    Note:
        - optimize_all_numeric=False: 既存のDtypeOptimizerと同等の挙動
        - optimize_all_numeric=True: preprocessing pipeline側のより積極的な最適化
    """
    dtypes: Dict[str, str] = {}

    for col in df.columns:
        col_series = _get_column_series(df, col)

        if col_series.dtype == "float64" or (
            optimize_all_numeric and pd.api.types.is_float_dtype(col_series)
        ):
            optimized = pd.to_numeric(col_series, downcast="float")
            if isinstance(optimized, pd.Series):
                if optimized.dtype != col_series.dtype:
                    dtypes[col] = str(optimized.dtype)
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
    """dtypeマップをDataFrameに適用してデータ型を変換する。

    dtypeマップに基づいて、DataFrameの各カラムのdtypeを一括変換します。
    メモリ使用量の削減を目的としています。

    Args:
        df: 変換対象のDataFrame。
        dtype_map: カラム名からdtype名へのマッピング。
            例: {"col1": "int32", "col2": "float32"}

    Returns:
        pd.DataFrame: dtype変換後のDataFrame。

    Raises:
        ValueError: 指定されたdtypeがpandasでサポートされていない場合。
            または変換中にデータ欠損が発生した場合。

    Note:
        DataFrameに存在しないカラムはスキップされます。
    """
    valid_dtypes = {k: v for k, v in dtype_map.items() if k in df.columns}
    return df.astype(valid_dtypes)


def optimize_dataframe_dtypes(
    df: pd.DataFrame,
    *,
    prefer_unsigned_integers: bool = False,
    optimize_all_numeric: bool = False,
) -> pd.DataFrame:
    """DataFrameをその場で最適化したdtypeに変換する。

    DataFrameのdtypeマップを構築し、適用してメモリ効率を向上させます。
    各カラムの実際のデータ範囲を分析し、最小限のビット数で表現可能な
    データ型を自動的に選択します。

    Args:
        df: 最適化対象のDataFrame。
        prefer_unsigned_integers: 符号なし整数を優先するかどうか（デフォルト: False）。
        optimize_all_numeric: 全ての数値型を最適化するかどうか（デフォルト: False）。

    Returns:
        pd.DataFrame: dtype最適化後のDataFrame。
            元のDataFrameは変更されず、新しいDataFrameが返されます。

    Raises:
        ValueError: 最適化中にデータ型の互換性問題が発生した場合。
    """
    dtype_map = build_optimized_dtype_map(
        df,
        prefer_unsigned_integers=prefer_unsigned_integers,
        optimize_all_numeric=optimize_all_numeric,
    )
    return apply_optimized_dtypes(df, dtype_map)


class DtypeOptimizer(BaseEstimator, TransformerMixin):
    """
    メモリ使用量を削減するためにデータ型を最適化するトランスフォーマー

    データの範囲を分析し、よりメモリ効率の高い代替データ型に変換します。
    （例: float64 -> float32, int64 -> int32/int16/int8）
    """

    def __init__(self):
        """
        DtypeOptimizerを初期化

        sklearn BaseEstimatorおよびTransformerMixinとの互換性を提供します。
        """

    def fit(self, X, y=None):
        """
        データの範囲を分析して最適なdtypeを決定してトランスフォーマーを適合

        入力データを分析して、各カラムの最適なdtypeを決定します。

        Args:
            X: 適合する入力データ（pandas.DataFrameまたはarray-like）
            y: 無視されます。sklearnパイプラインとの互換性のために存在

        Returns:
            self: 適合済みトランスフォーマー
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.dtypes_ = build_optimized_dtype_map(X)
        return self

    def transform(self, X):
        """
        データ型を最適化されたバージョンに変換

        fitで決定したdtypeマップを適用して、データ型を変換します。

        Args:
            X: 変換する入力データ（pandas.DataFrameまたはarray-like）

        Returns:
            pandas.DataFrame: 最適化されたデータ型のデータ
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return apply_optimized_dtypes(X, self.dtypes_)
