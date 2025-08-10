"""
指標計算用共通ユーティリティ

このモジュールはnumpy配列ベースの指標計算に必要な
共通機能を提供します。
"""

import logging
from functools import wraps
from typing import Union, cast

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
        if data.isna().any():
            logger.warning("入力データにNaN値が含まれています")
        if np.isinf(data).any():
            raise PandasTAError("入力データに無限大の値が含まれています")
    else:  # numpy配列の場合
        if np.any(np.isnan(data)):
            logger.warning("入力データにNaN値が含まれています")
        if np.any(np.isinf(data)):
            raise PandasTAError("入力データに無限大の値が含まれています")


def validate_multi_input(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int,
) -> None:
    """
    複数の価格データの検証（OHLC系指標用・pandas.Series対応版）

    Args:
        high: 高値データ（numpy配列またはpandas.Series）
        low: 安値データ（numpy配列またはpandas.Series）
        close: 終値データ（numpy配列またはpandas.Series）
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

    # 価格の論理的整合性チェック（pandas.Series対応）
    if isinstance(high, pd.Series) and isinstance(low, pd.Series):
        if (high < low).any():
            raise PandasTAError("高値が安値より低い箇所があります")
    else:
        # numpy配列の場合
        high_array = high.to_numpy() if isinstance(high, pd.Series) else high
        low_array = low.to_numpy() if isinstance(low, pd.Series) else low
        if np.any(high_array < low_array):
            raise PandasTAError("高値が安値より低い箇所があります")

    # 終値の範囲チェック
    if (
        isinstance(high, pd.Series)
        and isinstance(low, pd.Series)
        and isinstance(close, pd.Series)
    ):
        if ((close > high) | (close < low)).any():
            logger.warning("終値が高値・安値の範囲外の箇所があります")
    else:
        # numpy配列の場合
        high_array = high.to_numpy() if isinstance(high, pd.Series) else high
        low_array = low.to_numpy() if isinstance(low, pd.Series) else low
        close_array = close.to_numpy() if isinstance(close, pd.Series) else close
        if np.any((close_array > high_array) | (close_array < low_array)):
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


def validate_series_data(series: pd.Series, min_length: int) -> None:
    """
    pandas.Seriesデータの検証

    Args:
        series: 検証対象のpandas.Series
        min_length: 最小データ長

    Raises:
        PandasTAError: データが無効な場合
    """
    if series is None:
        raise PandasTAError("SeriesデータがNoneです")

    if not isinstance(series, pd.Series):
        raise PandasTAError(
            f"pandas.Seriesである必要があります。実際の型: {type(series)}"
        )

    if len(series) == 0:
        raise PandasTAError("Seriesデータが空です")

    if min_length <= 0:
        raise PandasTAError(f"最小長は正の整数である必要があります: {min_length}")

    if len(series) <= min_length:
        raise PandasTAError(f"データ長({len(series)})が最小長({min_length})以下です")

    # 数値型チェック
    if not pd.api.types.is_numeric_dtype(series):
        raise PandasTAError("Seriesデータは数値型である必要があります")

    # NaNや無限大の値をチェック
    if series.isna().all():
        # 全てNaNは明確にエラー
        raise PandasTAError("Seriesデータが全てNaNです")
    elif series.isna().any():
        logger.warning("SeriesデータにNaN値が含まれています")

    if np.isinf(series).any():
        raise PandasTAError("Seriesデータに無限大の値が含まれています")


def validate_indicator_parameters(*args, **kwargs) -> None:
    """
    インジケーターパラメータの検証

    Args:
        *args: 位置引数（通常は期間パラメータ）
        **kwargs: キーワード引数

    Raises:
        PandasTAError: パラメータが無効な場合
    """
    # 位置引数の検証（通常は期間パラメータ）
    for i, arg in enumerate(args):
        if isinstance(arg, (int, np.integer)):
            if arg <= 0:
                raise PandasTAError(
                    f"パラメータ{i+1}は正の値である必要があります: {arg}"
                )
        elif isinstance(arg, float):
            if arg <= 0:
                raise PandasTAError(
                    f"パラメータ{i+1}は正の値である必要があります: {arg}"
                )
            if not float(arg).is_integer():
                raise PandasTAError(f"パラメータ{i+1}は整数である必要があります: {arg}")
        elif arg is not None:
            # None以外の非数値パラメータは警告
            logger.warning(f"パラメータ{i+1}が数値ではありません: {type(arg)}")

    # キーワード引数の検証
    period_keys = {
        "length",
        "period",
        "timeperiod",
        "timeperiod1",
        "timeperiod2",
        "timeperiod3",
        "k",
        "d",
        "smooth_k",
        "fast",
        "medium",
        "slow",
        "k_period",
        "d_period",
        "period1",
        "period2",
        "period3",
    }
    for key, value in kwargs.items():
        if isinstance(value, (int, np.integer)):
            if value <= 0:
                raise PandasTAError(
                    f"パラメータ'{key}'は正の値である必要があります: {value}"
                )
        elif isinstance(value, float):
            if value <= 0:
                raise PandasTAError(
                    f"パラメータ'{key}'は正の値である必要があります: {value}"
                )
            if key in period_keys and not float(value).is_integer():
                raise PandasTAError(
                    f"パラメータ'{key}'は整数である必要があります: {value}"
                )


def normalize_data_for_trig(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    三角関数用のデータ正規化（-1から1の範囲にクリップ）

    Args:
        data: 正規化対象のデータ

    Returns:
        正規化されたnumpy配列

    Note:
        ASIN、ACOSなどの三角関数で使用するため、値を-1から1の範囲にクリップします。
    """
    if isinstance(data, pd.Series):
        array = data.to_numpy()
    elif isinstance(data, np.ndarray):
        array = data
    else:
        array = np.asarray(data, dtype=np.float64)

    # -1から1の範囲にクリップ
    return np.clip(array, -1.0, 1.0)


def ensure_series_minimal_conversion(data: Union[np.ndarray, pd.Series]) -> pd.Series:
    """
    最小限の型変換でpandas.Seriesを確保（型変換最小化版）

    Args:
        data: 入力データ

    Returns:
        pandas.Series

    Note:
        既にpandas.Seriesの場合は変換せずそのまま返します。
        numpy配列の場合のみpandas.Seriesに変換します。
    """
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, np.ndarray):
        return pd.Series(data, dtype=np.float64)
    else:
        # その他の型（list等）の場合
        return pd.Series(data, dtype=np.float64)


def ensure_numpy_minimal_conversion(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    最小限の型変換でnumpy配列を確保（型変換最小化版）

    Args:
        data: 入力データ

    Returns:
        numpy配列

    Note:
        既にnumpy配列の場合は変換せずそのまま返します。
        pandas.Seriesの場合のみnumpy配列に変換します。
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.Series):
        return data.to_numpy()
    else:
        # その他の型（list等）の場合
        return np.asarray(data, dtype=np.float64)


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
