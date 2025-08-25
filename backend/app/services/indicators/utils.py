"""
指標計算用共通ユーティリティ

このモジュールはnumpy配列ベースの指標計算に必要な
共通機能を提供します。
"""

import logging
from functools import wraps
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PandasTAError(Exception):
    """pandas-ta関連のエラー"""


def validate_input(data: Union[np.ndarray, pd.Series], period: int) -> None:
    """
    入力データの基本検証（numpy配列・pandas.Series対応版）

    Args:
        data: 検証対象のデータ（numpy配列またはpandas.Series）
        period: 期間パラメータ

    Raises:
        PandasTAError: 入力データが無効な場合
    """
    if data is None:
        raise PandasTAError("入力データがNoneです")

    if not isinstance(data, (np.ndarray, pd.Series)):
        raise PandasTAError(
            f"入力データはnumpy配列またはpandas.Seriesである必要があります。実際の型: {type(data)}"
        )

    if len(data) == 0:
        raise PandasTAError("入力データが空です")

    if period <= 0:
        raise PandasTAError(f"期間は正の整数である必要があります: {period}")

    if len(data) < period:
        raise PandasTAError(f"データ長({len(data)})が期間({period})より短いです")

    # NaNや無限大の値をチェック
    if isinstance(data, pd.Series):
        if bool(data.isna().any()):
            logger.warning("入力データにNaN値が含まれています")
        if np.isinf(data).any():
            raise PandasTAError("入力データに無限大の値が含まれています")
    else:  # numpy配列の場合
        if np.any(np.isnan(data)):
            logger.warning("入力データにNaN値が含まれています")
        if np.any(np.isinf(data)):
            raise PandasTAError("入力データに無限大の値が含まれています")


def handle_pandas_ta_errors(func):
    """
    pandas-taエラーハンドリングデコレーター

    重要な異常ケースのみをチェックし、パフォーマンスを重視。
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)

            # 重要な異常ケースのみチェック
            if result is None:
                raise PandasTAError(f"{func.__name__}: 計算結果がNoneです")

            # numpy配列の検証（簡素化）
            if isinstance(result, np.ndarray):
                if len(result) == 0:
                    raise PandasTAError(f"{func.__name__}: 計算結果が空です")
                # 全NaNチェック（重要）
                if len(result) > 0 and np.all(np.isnan(result)):
                    raise PandasTAError(f"{func.__name__}: 計算結果が全てNaNです")

            # tupleの場合（MACD等）
            elif isinstance(result, tuple):
                for i, arr in enumerate(result):
                    if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
                        raise PandasTAError(f"{func.__name__}: 結果[{i}]が無効です")

            return result

        except PandasTAError:
            # 既にPandasTAErrorの場合は再発生
            raise
        except Exception as e:
            # その他のエラーは簡潔に処理
            raise PandasTAError(f"{func.__name__} 計算エラー: {e}")

    return wrapper


def validate_numpy_input(
    data: Union[np.ndarray, pd.Series, list, float, int], min_length: int = 1
) -> np.ndarray:
    """入力データの検証とnumpy配列への変換"""
    # pandas.Seriesの場合はnumpy配列に変換
    if isinstance(data, pd.Series):
        data = data.to_numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data, dtype=float)

    # スカラー(ndim==0)はブロードキャスト前提で許容
    if isinstance(data, np.ndarray) and getattr(data, "ndim", 1) == 0:
        return data

    # 通常の長さチェック
    if len(data) < min_length:
        raise ValueError(f"データ長が不足: 必要{min_length}, 実際{len(data)}")

    return data


def validate_numpy_dual_input(
    data0: Union[np.ndarray, pd.Series], data1: Union[np.ndarray, pd.Series]
) -> None:
    """2つの入力データの長さ一致確認（スカラー許容）"""

    def _len_or_none(x):
        if isinstance(x, pd.Series):
            return len(x)
        if isinstance(x, np.ndarray):
            return None if x.ndim == 0 else len(x)
        try:
            return len(x)  # list等
        except Exception:
            return None

    len0 = _len_or_none(data0)
    len1 = _len_or_none(data1)

    # いずれかがスカラー（長さ不定）の場合はブロードキャスト前提で許容
    if len0 is None or len1 is None:
        return

    if len0 != len1:
        raise ValueError(f"データの長さが一致しません。Data0: {len0}, Data1: {len1}")
