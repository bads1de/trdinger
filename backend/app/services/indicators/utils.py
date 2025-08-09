"""
Ta-lib指標計算用共通ユーティリティ

このモジュールはnumpy配列ベースのTa-lib指標計算に必要な
共通機能を提供します。pandas Seriesの変換は一切行わず、
backtesting.pyとの完全な互換性を保ちます。
"""

import logging
from functools import wraps
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PandasTAError(Exception):
    """pandas-ta計算エラー"""


def validate_input(data: np.ndarray, period: int) -> None:
    """
    入力データの基本検証（numpy配列版）

    Args:
        data: 検証対象のnumpy配列
        period: 期間パラメータ

    Raises:
        PandasTAError: 入力データが無効な場合
    """
    if data is None:
        raise PandasTAError("入力データがNoneです")

    if not isinstance(data, np.ndarray):
        raise PandasTAError(
            f"入力データはnumpy配列である必要があります。実際の型: {type(data)}"
        )

    if len(data) == 0:
        raise PandasTAError("入力データが空です")

    if period <= 0:
        raise PandasTAError(f"期間は正の整数である必要があります: {period}")

    if len(data) < period:
        raise PandasTAError(f"データ長({len(data)})が期間({period})より短いです")

    # NaNや無限大の値をチェック
    if np.any(np.isnan(data)):
        logger.warning("入力データにNaN値が含まれています")

    if np.any(np.isinf(data)):
        raise PandasTAError("入力データに無限大の値が含まれています")


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
        PandasTAError: 入力データが無効な場合
    """
    # 各データの基本検証
    validate_input(high, period)
    validate_input(low, period)
    validate_input(close, period)

    # データ長の一致確認
    if not (len(high) == len(low) == len(close)):
        raise PandasTAError(
            f"価格データの長さが一致しません。High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
        )

    # 価格の論理的整合性チェック
    if np.any(high < low):
        raise PandasTAError("高値が安値より低い箇所があります")

    if np.any((close > high) | (close < low)):
        logger.warning("終値が高値・安値の範囲外の箇所があります")


def handle_pandas_ta_errors(func):
    """
    pandas-taエラーハンドリングデコレーター

    pandas-ta関数の実行時エラーをキャッチし、
    適切なログ出力とエラーメッセージを提供します。
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)

            # 結果の基本検証
            if result is None:
                raise PandasTAError(f"{func.__name__}: 計算結果がNoneです")

            # numpy配列の場合の検証
            if isinstance(result, np.ndarray):
                if len(result) == 0:
                    raise PandasTAError(f"{func.__name__}: 計算結果が空の配列です")

                # 全てNaNの場合は警告
                if np.all(np.isnan(result)):
                    logger.warning(f"{func.__name__}: 計算結果が全てNaNです")

            # tupleの場合（MACD、Bollinger Bandsなど）
            elif isinstance(result, tuple):
                for i, arr in enumerate(result):
                    if arr is None or len(arr) == 0:
                        raise PandasTAError(
                            f"{func.__name__}: 計算結果のインデックス{i}が無効です"
                        )

            return result

        except Exception as e:
            # Ta-lib固有のエラーかどうかを判定
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in ["pandas", "ta", "period", "length", "array"]
            ):
                logger.error(f"{func.__name__} pandas-ta計算エラー: {e}")
                raise PandasTAError(f"{func.__name__} 計算失敗: {e}")
            else:
                # 予期しないエラー
                logger.error(f"{func.__name__} 予期しないエラー: {e}")
                raise PandasTAError(f"{func.__name__} 予期しないエラー: {e}")

    return wrapper


def ensure_numpy_array(data: Union[np.ndarray, list, "pd.Series"]) -> np.ndarray:
    """
    データをnumpy配列に変換（オートストラテジー最適化）

    Args:
        data: 変換対象のデータ

    Returns:
        float64型のnumpy配列

    Note:
        pandas-taはfloat64型を要求するため、必ずfloat64に変換します。
    """
    if data is None:
        raise PandasTAError("データがNoneです")

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

        # pandas-ta用にfloat64に変換
        if array.dtype != np.float64:
            return array.astype(np.float64)
        return array

    except (ValueError, TypeError) as e:
        raise PandasTAError(f"データをnumpy配列に変換できません: {e}")


def format_indicator_result(
    result: Union[np.ndarray, tuple],
    indicator_name: str | None = None,
) -> Union[np.ndarray, tuple]:
    """
    指標計算結果のフォーマット（現在はプレースホルダー）

    Args:
        result: 計算結果
        indicator_name: 任意の指標名（ログ・将来の拡張用、未使用）

    Returns:
        フォーマット済みの結果
    """
    # 現在は特に処理を行わず、そのまま返す
    return result


def normalize_data_for_trig(data: np.ndarray) -> np.ndarray:
    """
    三角関数（ASIN, ACOS）の入力用にデータを[-1, 1]の範囲に正規化します。

    Args:
        data: 正規化対象のnumpy配列

    Returns:
        正規化されたnumpy配列
    """
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    # 全て同じ値の場合（ゼロ除算を避ける）
    if min_val == max_val:
        # 範囲内に収まるように0または適切な値を返す
        return np.full_like(data, 0.0)

    # Min-Maxスケーリングで[0, 1]に正規化し、その後[-1, 1]に変換
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1

    # 計算誤差により範囲外になる可能性を考慮し、クリッピング
    return np.clip(normalized_data, -1.0, 1.0)


# 後方互換性のためのエイリアス
TALibError = PandasTAError
handle_talib_errors = handle_pandas_ta_errors
