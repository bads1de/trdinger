"""
指標計算用共通ユーティリティ

このモジュールはnumpy配列ベースの指標計算に必要な
共通機能を提供します。
"""

import logging
from functools import wraps


import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PandasTAError(Exception):
    """pandas-ta関連のエラー"""


def validate_input(data: pd.Series, period: int) -> None:
    """
    入力データの基本検証（pandas.Series専用）

    Args:
        data: 検証対象のデータ（pandas.Series）
        period: 期間パラメータ

    Raises:
        PandasTAError: 入力データが無効な場合
    """
    if data is None:
        raise PandasTAError("入力データがNoneです")

    if not isinstance(data, pd.Series):
        raise PandasTAError(
            f"入力データはpandas.Seriesである必要があります。実際の型: {type(data)}"
        )

    if len(data) == 0:
        raise PandasTAError("入力データが空です")

    if period <= 0:
        raise PandasTAError(f"期間は正の整数である必要があります: {period}")

    if len(data) < period:
        raise PandasTAError(f"データ長({len(data)})が期間({period})より短いです")

    # NaNや無限大の値をチェック (pandas.Series専用)
    if bool(data.isna().any()):
        logger.warning("入力データにNaN値が含まれています")
    if np.isinf(data).any():
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
        except (TypeError, ValueError):
            # バリデーションエラーは再発生
            raise
        except Exception as e:
            # その他のエラーは簡潔に処理
            raise PandasTAError(f"{func.__name__} 計算エラー: {e}")

    return wrapper
