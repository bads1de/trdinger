"""
Ta-lib指標計算用共通ユーティリティ（オートストラテジー最適化版）

このモジュールはnumpy配列ベースのTa-lib指標計算に必要な
共通機能を提供します。pandas Seriesの変換は一切行わず、
backtesting.pyとの完全な互換性を保ちます。
"""

import logging
import numpy as np
import pandas as pd
from functools import wraps
from typing import Union

logger = logging.getLogger(__name__)


class TALibError(Exception):
    """Ta-lib計算エラー"""

    pass


def validate_input(data: np.ndarray, period: int) -> None:
    """
    入力データの基本検証（numpy配列版）

    Args:
        data: 検証対象のnumpy配列
        period: 期間パラメータ

    Raises:
        TALibError: 入力データが無効な場合
    """
    if data is None:
        raise TALibError("入力データがNoneです")

    if not isinstance(data, np.ndarray):
        raise TALibError(
            f"入力データはnumpy配列である必要があります。実際の型: {type(data)}"
        )

    if len(data) == 0:
        raise TALibError("入力データが空です")

    if period <= 0:
        raise TALibError(f"期間は正の整数である必要があります: {period}")

    if len(data) < period:
        raise TALibError(f"データ長({len(data)})が期間({period})より短いです")

    # NaNや無限大の値をチェック
    if np.any(np.isnan(data)):
        logger.warning("入力データにNaN値が含まれています")

    if np.any(np.isinf(data)):
        raise TALibError("入力データに無限大の値が含まれています")


def validate_multi_input(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> None:
    """
    複数の価格データの検証（OHLC系指標用）

    Args:
        high: 高値データ
        low: 安値データ
        close: 終値データ
        period: 期間パラメータ

    Raises:
        TALibError: 入力データが無効な場合
    """
    # 各データの基本検証
    validate_input(high, period)
    validate_input(low, period)
    validate_input(close, period)

    # データ長の一致確認
    if not (len(high) == len(low) == len(close)):
        raise TALibError(
            f"価格データの長さが一致しません。High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
        )

    # 価格の論理的整合性チェック
    if np.any(high < low):
        raise TALibError("高値が安値より低い箇所があります")

    if np.any((close > high) | (close < low)):
        logger.warning("終値が高値・安値の範囲外の箇所があります")


def handle_talib_errors(func):
    """
    Ta-libエラーハンドリングデコレーター

    Ta-lib関数の実行時エラーをキャッチし、
    適切なログ出力とエラーメッセージを提供します。
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)

            # 結果の基本検証
            if result is None:
                raise TALibError(f"{func.__name__}: 計算結果がNoneです")

            # numpy配列の場合の検証
            if isinstance(result, np.ndarray):
                if len(result) == 0:
                    raise TALibError(f"{func.__name__}: 計算結果が空の配列です")

                # 全てNaNの場合は警告
                if np.all(np.isnan(result)):
                    logger.warning(f"{func.__name__}: 計算結果が全てNaNです")

            # tupleの場合（MACD、Bollinger Bandsなど）
            elif isinstance(result, tuple):
                for i, arr in enumerate(result):
                    if arr is None or len(arr) == 0:
                        raise TALibError(
                            f"{func.__name__}: 計算結果のインデックス{i}が無効です"
                        )

            return result

        except Exception as e:
            # Ta-lib固有のエラーかどうかを判定
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in ["talib", "period", "length", "array"]
            ):
                logger.error(f"{func.__name__} Ta-lib計算エラー: {e}")
                raise TALibError(f"{func.__name__} 計算失敗: {e}")
            else:
                # 予期しないエラー
                logger.error(f"{func.__name__} 予期しないエラー: {e}")
                raise TALibError(f"{func.__name__} 予期しないエラー: {e}")

    return wrapper


def ensure_numpy_array(data: Union[np.ndarray, list, "pd.Series"]) -> np.ndarray:
    """
    データをnumpy配列に変換（オートストラテジー最適化）

    Args:
        data: 変換対象のデータ

    Returns:
        float64型のnumpy配列

    Note:
        Ta-libはfloat64型を要求するため、必ずfloat64に変換します。
    """
    if data is None:
        raise TALibError("データがNoneです")

    try:
        # pandas Seriesの場合
        if isinstance(data, pd.Series):
            array = data.to_numpy()
        # 既にnumpy配列の場合
        elif isinstance(data, np.ndarray):
            array = data
        # listやその他のシーケンス型の場合
        else:
            array = np.asarray(data, dtype=np.float64)

        # Ta-lib用にfloat64に変換
        if array.dtype != np.float64:
            return array.astype(np.float64)
        return array

    except (ValueError, TypeError) as e:
        raise TALibError(f"データをnumpy配列に変換できません: {e}")


def log_indicator_calculation(
    indicator_name: str, parameters: dict, data_length: int
) -> None:
    """
    指標計算のログ出力

    Args:
        indicator_name: 指標名
        parameters: パラメータ辞書
        data_length: データ長
    """
    # logger.debug(
    #     f"指標計算開始: {indicator_name}, パラメータ: {parameters}, データ長: {data_length}"
    # )
    pass


def format_indicator_result(
    result: Union[np.ndarray, tuple], indicator_name: str
) -> Union[np.ndarray, tuple]:
    """
    指標計算結果のフォーマット

    Args:
        result: 計算結果
        indicator_name: 指標名

    Returns:
        フォーマット済みの結果
    """
    if isinstance(result, np.ndarray):
        # NaNの数をログ出力
        nan_count = np.sum(np.isnan(result))
        if nan_count > 0:
            # logger.debug(f"{indicator_name}: {nan_count}個のNaN値があります")
            pass

        return result

    elif isinstance(result, tuple):
        # tupleの各要素をチェック
        formatted_results = []
        for i, arr in enumerate(result):
            if isinstance(arr, np.ndarray):
                nan_count = np.sum(np.isnan(arr))
                if nan_count > 0:
                    # logger.debug(
                    #     f"{indicator_name}[{i}]: {nan_count}個のNaN値があります"
                    # )
                    pass
            formatted_results.append(arr)

        return tuple(formatted_results)

    return result
