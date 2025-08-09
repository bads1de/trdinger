"""
pandas-ta用ユーティリティ関数

TA-libからpandas-taへの移行をサポートするユーティリティ関数群
"""

import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Union, Optional, Tuple, cast
from functools import wraps

logger = logging.getLogger(__name__)


class PandasTAError(Exception):
    """pandas-ta関連のエラー"""

    pass


def ensure_pandas_series(
    data: Union[np.ndarray, pd.Series], index: Optional[pd.Index] = None
) -> pd.Series:
    """
    データをpandas Seriesに変換

    Args:
        data: 入力データ（numpy配列またはpandas Series）
        index: インデックス（numpy配列の場合）

    Returns:
        pandas Series
    """
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, np.ndarray):
        if index is None:
            index = pd.RangeIndex(len(data))
        return pd.Series(data, index=index)
    else:
        raise PandasTAError(f"サポートされていないデータ型: {type(data)}")


def ensure_numpy_array_from_series(series: pd.Series) -> np.ndarray:
    """
    pandas SeriesをNumPy配列に変換（backtesting.py互換性のため）

    Args:
        series: pandas Series

    Returns:
        numpy配列
    """
    return series.values


def validate_pandas_input(data: pd.Series, min_length: int = 1) -> None:
    """
    pandas Series入力データの検証

    Args:
        data: 入力データ
        min_length: 最小データ長

    Raises:
        PandasTAError: 入力データが無効な場合
    """
    if not isinstance(data, pd.Series):
        raise PandasTAError(f"pandas Seriesが必要です。実際の型: {type(data)}")

    if len(data) < min_length:
        raise PandasTAError(
            f"データ長が不足しています。必要: {min_length}, 実際: {len(data)}"
        )

    if data.isna().all():
        raise PandasTAError("全てのデータがNaNです")


def validate_pandas_multi_input(*series: pd.Series, min_length: int = 1) -> None:
    """
    複数のpandas Series入力データの検証

    Args:
        *series: 複数のpandas Series
        min_length: 最小データ長

    Raises:
        PandasTAError: 入力データが無効な場合
    """
    if not series:
        raise PandasTAError("少なくとも1つのSeriesが必要です")

    # 各Seriesの基本検証
    for i, s in enumerate(series):
        try:
            validate_pandas_input(s, min_length)
        except PandasTAError as e:
            raise PandasTAError(f"Series {i}: {str(e)}")

    # 長さの一致確認
    lengths = [len(s) for s in series]
    if len(set(lengths)) > 1:
        raise PandasTAError(f"Seriesの長さが一致しません: {lengths}")

    # インデックスの一致確認（警告のみ）
    first_index = series[0].index
    for i, s in enumerate(series[1:], 1):
        if not first_index.equals(s.index):
            logger.warning(f"Series {i}のインデックスが一致しません")


def handle_pandas_ta_errors(func):
    """
    pandas-taエラーハンドリングデコレーター

    pandas-ta関数の実行時エラーをキャッチし、
    適切なログ出力とエラーメッセージを提供します。
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            func_name = func.__name__
            logger.error(f"pandas-ta関数 {func_name} でエラーが発生しました: {str(e)}")

            # 具体的なエラータイプに応じた処理
            if isinstance(e, (ValueError, TypeError)):
                raise PandasTAError(f"{func_name}: 入力パラメータエラー - {str(e)}")
            elif isinstance(e, KeyError):
                raise PandasTAError(f"{func_name}: データアクセスエラー - {str(e)}")
            else:
                raise PandasTAError(f"{func_name}: 予期しないエラー - {str(e)}")

    return wrapper


def format_pandas_ta_result(
    result: Union[pd.Series, pd.DataFrame], indicator_name: str
) -> Union[pd.Series, pd.DataFrame]:
    """
    pandas-ta結果のフォーマット

    Args:
        result: pandas-taの計算結果
        indicator_name: 指標名

    Returns:
        フォーマットされた結果
    """
    if result is None:
        raise PandasTAError(f"{indicator_name}: 計算結果がNoneです")

    if isinstance(result, pd.Series):
        # NaN値の処理
        if result.isna().all():
            logger.warning(f"{indicator_name}: 全ての値がNaNです")
        return result

    elif isinstance(result, pd.DataFrame):
        # DataFrameの場合、各列をチェック
        for col in result.columns:
            if result[col].isna().all():
                logger.warning(f"{indicator_name}.{col}: 全ての値がNaNです")
        return result

    else:
        raise PandasTAError(
            f"{indicator_name}: サポートされていない結果型: {type(result)}"
        )


def convert_to_numpy_for_backtesting(
    result: Union[pd.Series, pd.DataFrame],
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    pandas-taの結果をbacktesting.py互換のnumpy配列に変換

    Args:
        result: pandas-taの計算結果

    Returns:
        numpy配列または複数のnumpy配列のタプル
    """
    if isinstance(result, pd.Series):
        return result.values
    elif isinstance(result, pd.DataFrame):
        if len(result.columns) == 1:
            return result.iloc[:, 0].values
        else:
            return tuple(result[col].values for col in result.columns)
    else:
        raise PandasTAError(f"サポートされていない結果型: {type(result)}")


# pandas-ta関数のラッパー関数群


@handle_pandas_ta_errors
def pandas_ta_sma(
    data: Union[np.ndarray, pd.Series], length: int, index: Optional[pd.Index] = None
) -> np.ndarray:
    """
    pandas-ta版SMA（backtesting.py互換）

    Args:
        data: 価格データ
        length: 期間
        index: インデックス（numpy配列の場合）

    Returns:
        SMA値のnumpy配列
    """
    # pandas Seriesに変換
    series = ensure_pandas_series(data, index)
    validate_pandas_input(series, length)

    # pandas-taでSMA計算
    result = ta.sma(series, length=length)
    result = format_pandas_ta_result(result, "SMA")

    # numpy配列に変換してbacktesting.py互換性を保つ
    return ensure_numpy_array_from_series(result)


@handle_pandas_ta_errors
def pandas_ta_ema(
    data: Union[np.ndarray, pd.Series], length: int, index: Optional[pd.Index] = None
) -> np.ndarray:
    """
    pandas-ta版EMA（backtesting.py互換）

    Args:
        data: 価格データ
        length: 期間
        index: インデックス（numpy配列の場合）

    Returns:
        EMA値のnumpy配列
    """
    # pandas Seriesに変換
    series = ensure_pandas_series(data, index)
    validate_pandas_input(series, length)

    # pandas-taでEMA計算
    result = ta.ema(series, length=length)
    result = format_pandas_ta_result(result, "EMA")

    # numpy配列に変換してbacktesting.py互換性を保つ
    return ensure_numpy_array_from_series(result)


@handle_pandas_ta_errors
def pandas_ta_rsi(
    data: Union[np.ndarray, pd.Series],
    length: int = 14,
    index: Optional[pd.Index] = None,
) -> np.ndarray:
    """
    pandas-ta版RSI（backtesting.py互換）

    Args:
        data: 価格データ
        length: 期間
        index: インデックス（numpy配列の場合）

    Returns:
        RSI値のnumpy配列
    """
    # pandas Seriesに変換
    series = ensure_pandas_series(data, index)
    validate_pandas_input(series, length + 1)  # RSIは期間+1が必要

    # pandas-taでRSI計算
    result = ta.rsi(series, length=length)
    result = format_pandas_ta_result(result, "RSI")

    # numpy配列に変換してbacktesting.py互換性を保つ
    return ensure_numpy_array_from_series(result)


@handle_pandas_ta_errors
def pandas_ta_macd(
    data: Union[np.ndarray, pd.Series],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    index: Optional[pd.Index] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pandas-ta版MACD（backtesting.py互換）

    Args:
        data: 価格データ
        fast: 高速期間
        slow: 低速期間
        signal: シグナル期間
        index: インデックス（numpy配列の場合）

    Returns:
        (MACD, Signal, Histogram)のタプル
    """
    # pandas Seriesに変換
    series = ensure_pandas_series(data, index)
    validate_pandas_input(series, slow + signal)

    # pandas-taでMACD計算
    result = ta.macd(series, fast=fast, slow=slow, signal=signal)
    result = format_pandas_ta_result(result, "MACD")

    # 複数の戻り値をnumpy配列のタプルに変換
    if isinstance(result, pd.DataFrame):
        macd_col = f"MACD_{fast}_{slow}_{signal}"
        signal_col = f"MACDs_{fast}_{slow}_{signal}"
        hist_col = f"MACDh_{fast}_{slow}_{signal}"

        return (
            result[macd_col].values,
            result[signal_col].values,
            result[hist_col].values,
        )
    else:
        raise PandasTAError("MACDの結果がDataFrameではありません")


@handle_pandas_ta_errors
def pandas_ta_atr(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    length: int = 14,
    index: Optional[pd.Index] = None,
) -> np.ndarray:
    """
    pandas-ta版ATR（backtesting.py互換）

    Args:
        high: 高値データ
        low: 安値データ
        close: 終値データ
        length: 期間
        index: インデックス（numpy配列の場合）

    Returns:
        ATR値のnumpy配列
    """
    # pandas Seriesに変換
    high_series = ensure_pandas_series(high, index)
    low_series = ensure_pandas_series(low, index)
    close_series = ensure_pandas_series(close, index)

    validate_pandas_multi_input(
        high_series, low_series, close_series, min_length=length
    )

    # pandas-taでATR計算
    result = ta.atr(high=high_series, low=low_series, close=close_series, length=length)
    result = format_pandas_ta_result(result, "ATR")

    # numpy配列に変換してbacktesting.py互換性を保つ
    return ensure_numpy_array_from_series(result)


@handle_pandas_ta_errors
def pandas_ta_bbands(
    data: Union[np.ndarray, pd.Series],
    length: int = 20,
    std: float = 2.0,
    index: Optional[pd.Index] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pandas-ta版Bollinger Bands（backtesting.py互換）

    Args:
        data: 価格データ
        length: 期間
        std: 標準偏差の倍数
        index: インデックス（numpy配列の場合）

    Returns:
        (Upper Band, Middle Band, Lower Band)のタプル
    """
    # pandas Seriesに変換
    series = ensure_pandas_series(data, index)
    validate_pandas_input(series, length)

    # pandas-taでBollinger Bands計算
    result = ta.bbands(series, length=length, std=std)
    result = format_pandas_ta_result(result, "BBANDS")

    # 複数の戻り値をnumpy配列のタプルに変換
    if isinstance(result, pd.DataFrame):
        upper_col = f"BBU_{length}_{std}"
        middle_col = f"BBM_{length}_{std}"
        lower_col = f"BBL_{length}_{std}"

        return (
            result[upper_col].values,
            result[middle_col].values,
            result[lower_col].values,
        )
    else:
        raise PandasTAError("Bollinger Bandsの結果がDataFrameではありません")
