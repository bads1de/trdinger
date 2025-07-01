"""
データ変換ユーティリティ

汎用的なデータ変換機能を提供するモジュール。
backtesting.pyの_Arrayオブジェクト、pandas.Series、numpy配列、リストなどの
相互変換を統一的に処理します。
"""

import logging

import pandas as pd
import numpy as np
from typing import Union, Any, Optional


logger = logging.getLogger(__name__)


class DataConversionError(Exception):
    """データ変換エラー"""

    pass


def ensure_series(
    data: Union[pd.Series, list, np.ndarray, Any],
    raise_on_error: bool = True,
    name: Optional[str] = None,
) -> pd.Series:
    """
    データをpandas.Seriesに変換

    Args:
        data: 入力データ（pandas.Series, list, numpy.ndarray, backtesting._Array等）
        raise_on_error: エラー時に例外を発生させるかどうか
        name: 作成するSeriesの名前

    Returns:
        pandas.Series

    Raises:
        DataConversionError: サポートされていないデータ型の場合（raise_on_error=Trueの時）
    """
    try:
        # 既にpandas.Seriesの場合
        if isinstance(data, pd.Series):
            if name is not None and name != data.name:
                result = data.copy()
                result.name = name
                return result
            return data

        # list, numpy.ndarray（valuesより先にチェック）
        if isinstance(data, (list, np.ndarray)):
            return pd.Series(data, name=name)

        # backtesting.pyの_Arrayオブジェクト（_data属性を持つ）
        if hasattr(data, "_data"):
            return pd.Series(data._data, name=name)

        # valuesアトリビュートを持つオブジェクト（pandas.DataFrame等、ただし辞書は除外）
        if hasattr(data, "values") and not isinstance(data, dict):
            return pd.Series(data.values, name=name)

        # その他のデータ型（スカラー値等）
        if np.isscalar(data):
            return pd.Series([data], name=name)

        # 配列風のオブジェクトを試行（文字列、辞書、セットは除外）
        if not isinstance(data, (str, dict, set)):
            try:
                # 反復可能かチェック
                iter(data)
                return pd.Series(data, name=name)
            except (ValueError, TypeError):
                pass

        # サポートされていないデータ型
        if raise_on_error:
            raise DataConversionError(f"サポートされていないデータ型: {type(data)}")
        else:
            logger.error(f"データ変換エラー: サポートされていないデータ型 {type(data)}")
            return pd.Series([], name=name)

    except Exception as e:
        if raise_on_error:
            if isinstance(e, DataConversionError):
                raise
            else:
                raise DataConversionError(f"データ変換エラー: {e}")
        else:
            logger.error(f"データ変換エラー: {e}")
            return pd.Series([], name=name)


def ensure_numeric_series(
    data: Union[pd.Series, list, np.ndarray, Any],
    raise_on_error: bool = True,
    name: Optional[str] = None,
) -> pd.Series:
    """
    データを数値型のpandas.Seriesに変換

    Args:
        data: 入力データ
        raise_on_error: エラー時に例外を発生させるかどうか
        name: 作成するSeriesの名前

    Returns:
        数値型のpandas.Series

    Raises:
        DataConversionError: 変換に失敗した場合（raise_on_error=Trueの時）
    """
    try:
        series = ensure_series(data, raise_on_error=raise_on_error, name=name)

        # 数値型に変換
        numeric_series = pd.to_numeric(
            series, errors="coerce" if not raise_on_error else "raise"
        )

        if raise_on_error and numeric_series.isna().any():
            raise DataConversionError("数値に変換できない値が含まれています")

        return numeric_series

    except Exception as e:
        if raise_on_error:
            if isinstance(e, DataConversionError):
                raise
            else:
                raise DataConversionError(f"数値変換エラー: {e}")
        else:
            logger.error(f"数値変換エラー: {e}")
            return pd.Series([], dtype=float, name=name)


def validate_series_length(*series: pd.Series) -> None:
    """
    複数のSeriesの長さが一致することを検証

    Args:
        *series: 検証対象のSeries

    Raises:
        DataConversionError: Series長が一致しない場合
    """
    if not series:
        raise DataConversionError("検証対象のSeriesが指定されていません")

    lengths = [len(s) for s in series]
    if not all(length == lengths[0] for length in lengths):
        raise DataConversionError(f"Series長が一致しません: {lengths}")

    if lengths[0] == 0:
        raise DataConversionError("入力データが空です")


def validate_series_data(series: pd.Series, min_length: int = 1) -> None:
    """
    Seriesデータの妥当性を検証

    Args:
        series: 検証対象のSeries
        min_length: 最小長

    Raises:
        DataConversionError: データが無効な場合
    """
    if series is None:
        raise DataConversionError("Seriesがnullです")

    if len(series) < min_length:
        raise DataConversionError(
            f"データ長({len(series)})が最小長({min_length})より短いです"
        )

    if series.isna().all():
        raise DataConversionError("全ての値がNaNです")


def create_series_with_index(
    data: Union[list, np.ndarray],
    index: Optional[pd.Index] = None,
    name: Optional[str] = None,
) -> pd.Series:
    """
    指定されたインデックスでSeriesを作成

    Args:
        data: データ配列
        index: インデックス（Noneの場合はデフォルトインデックス）
        name: Series名

    Returns:
        pandas.Series

    Raises:
        DataConversionError: データとインデックスの長さが一致しない場合
    """
    try:
        if index is not None and len(data) != len(index):
            raise DataConversionError(
                f"データ長({len(data)})とインデックス長({len(index)})が一致しません"
            )

        return pd.Series(data, index=index, name=name)

    except Exception as e:
        raise DataConversionError(f"Series作成エラー: {e}")


def safe_array_conversion(data: Any) -> np.ndarray:
    """
    データを安全にnumpy配列に変換

    Args:
        data: 変換対象のデータ

    Returns:
        numpy.ndarray

    Raises:
        DataConversionError: 変換に失敗した場合
    """
    try:
        if isinstance(data, np.ndarray):
            return data

        if isinstance(data, pd.Series):
            return data.to_numpy()

        if hasattr(data, "_data"):
            return np.array(data._data)

        if hasattr(data, "values"):
            if hasattr(data, "to_numpy"):
                return data.to_numpy()
            return np.array(data.values)

        return np.array(data)

    except Exception as e:
        raise DataConversionError(f"numpy配列変換エラー: {e}")


# 後方互換性のためのエイリアス
def convert_to_series(data, raise_on_error: bool = False) -> pd.Series:
    """
    後方互換性のためのエイリアス

    Args:
        data: 変換対象のデータ
        raise_on_error: エラー時に例外を発生させるかどうか

    Returns:
        pandas.Series
    """
    return ensure_series(data, raise_on_error=raise_on_error)


def _ensure_series(data: Union[pd.Series, list, np.ndarray]) -> pd.Series:
    """
    後方互換性のためのエイリアス（BaseAdapter._ensure_series互換）

    Args:
        data: 入力データ

    Returns:
        pandas.Series

    Raises:
        DataConversionError: サポートされていないデータ型の場合
    """
    return ensure_series(data, raise_on_error=True)
