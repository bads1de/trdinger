"""
データバリデーション・エラーハンドリングモジュール

テクニカル指標計算に必要なバリデーション機能を統合的に提供します。

主な機能:
- データ長検証と最小要件チェック
- 入力データのバリデーション
- エラーハンドリングデコレーター
- NaN結果生成
"""

import logging
from functools import wraps
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# カスタム例外
# =============================================================================


class PandasTAError(Exception):
    """pandas-ta関連のエラー"""


# =============================================================================
# データ長検証
# =============================================================================


def get_param_value(params: Dict[str, Any], keys: list, default: Any) -> Any:
    """パラメータ名がlengthまたはwindowの場合の値取得をサポート"""
    for key in keys:
        if key in params:
            return params[key]
    return default


def get_minimum_data_length(indicator_type: str, params: Dict[str, Any]) -> int:
    """
    指標の種類とパラメータから最小必要データ長を取得

    Args:
        indicator_type: 指標タイプ
        params: パラメータ辞書

    Returns:
        最小必要データ長
    """
    from .config.indicator_config import indicator_registry

    config = indicator_registry.get_indicator_config(indicator_type.upper())
    if config and config.min_length_func:
        return config.min_length_func(params)

    # フォールバック：デフォルト値 - lengthまたはwindowパラメータをサポート
    if config and config.default_values:
        length_value = get_param_value(config.default_values, ["length", "window"], 14)
        return length_value

    return 1  # 最低1つのデータ点


def get_absolute_minimum_length(indicator_type: str) -> int:
    """
    各指標の絶対的最小データ長を取得
    """
    from .config.indicator_config import indicator_registry

    config = indicator_registry.get_indicator_config(indicator_type.upper())
    if (
        config
        and hasattr(config, "absolute_min_length")
        and config.absolute_min_length is not None
    ):
        return int(config.absolute_min_length)
    return 1


def validate_data_length_with_fallback(
    df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
) -> Tuple[bool, int]:
    """
    データ長検証を強化し、フォールバック可能な最小データ長を返す

    Args:
        df: OHLCV価格データ
        indicator_type: 指標タイプ
        params: パラメータ辞書

    Returns:
        (データ長が十分かどうか, フォールバック可能な最小データ長)
    """
    standard_length = get_minimum_data_length(indicator_type, params) or 1
    absolute_minimum = get_absolute_minimum_length(indicator_type.upper()) or 1

    # パラメータに基づく長さと絶対的な最小長さの大きい方を採用
    required_length = max(int(standard_length), int(absolute_minimum))

    data_length = len(df)

    if data_length >= required_length:
        return True, required_length

    # フォールバック処理のための緩和された最小長を計算
    min_required = max(int(absolute_minimum), int(standard_length) // 3)

    if data_length >= min_required:
        logger.info(
            "データ長不足時は、この関数でフォールバック加工を実行。"
            f"{indicator_type}: 必要データ長 {required_length} が {data_length} 不足のため、"
            f"最小データ長 {min_required} にフォールしNaNフィルタを適用"
        )
        return True, data_length

    logger.warning(
        "この関数で運用データでの警告出力を強化。"
        f"{indicator_type}: 必要なデータ長 {required_length} 以上、"
        f"最低データ長 {min_required} が {data_length} 不足のためNaNフィルタを適用"
    )
    return False, min_required


# =============================================================================
# NaN結果生成
# =============================================================================


def create_nan_result(df: pd.DataFrame, indicator_type: str) -> np.ndarray:
    """
    データ長不足時のNaN結果生成

    Args:
        df: 元のデータフレーム
        indicator_type: 指標タイプ

    Returns:
        NaN配列
    """
    from .config.indicator_config import indicator_registry

    config = indicator_registry.get_indicator_config(indicator_type.upper())
    data_length = len(df)

    if not config:
        return np.full(data_length, np.nan)

    if config.returns == "single":
        return np.full(data_length, np.nan)
    else:
        return_cols = config.return_cols or ["Result"]
        nan_result = np.full((data_length, len(return_cols)), np.nan)
        return nan_result


# =============================================================================
# 入力値バリデーション
# =============================================================================


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


def validate_series_params(
    data: pd.Series, length: int = None, min_data_length: int = 0
) -> Optional[pd.Series]:
    """
    指標計算用のパラメータ検証（共通化用）

    Args:
        data: 入力データ
        length: 期間（オプション）
        min_data_length: 最小必要データ長（オプション）

    Returns:
        pd.Series: データが空または不足している場合のNaNシリーズ（計算不要）
        None: 検証OK、計算続行

    Raises:
        TypeError: データ型が無効な場合
        ValueError: 期間が無効な場合
    """
    if not isinstance(data, pd.Series):
        raise TypeError("data must be pandas Series")

    if length is not None and length <= 0:
        raise ValueError(f"length must be positive: {length}")

    if len(data) == 0:
        return pd.Series(np.full(0, np.nan), index=data.index)

    if min_data_length > 0 and len(data) < min_data_length:
        return pd.Series(np.full(len(data), np.nan), index=data.index)

    return None


def validate_multi_series_params(
    series_dict: dict,
    length: int = None,
    min_data_length: int = 0,
) -> Optional[pd.Series]:
    """
    複数のSeriesパラメータを検証（共通化用）

    Args:
        series_dict: 検証する名前付きシリーズの辞書。例: {"high": high, "low": low}
        length: 期間（オプション）
        min_data_length: 最小必要データ長（オプション）

    Returns:
        pd.Series: データが空または不足している場合のNaNシリーズ（計算不要）
        None: 検証OK、計算続行

    Raises:
        TypeError: データ型が無効な場合
        ValueError: 期間が無効な場合、またはシリーズ長が不一致な場合
    """
    if not series_dict:
        raise ValueError("series_dict cannot be empty")

    first_series = None
    first_name = None

    for name, series in series_dict.items():
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be pandas Series")

        if first_series is None:
            first_series = series
            first_name = name
        elif len(series) != len(first_series):
            raise ValueError(
                f"All series must have the same length. "
                f"{first_name}={len(first_series)}, {name}={len(series)}"
            )

    if length is not None and length <= 0:
        raise ValueError(f"length must be positive: {length}")

    if first_series is not None and len(first_series) == 0:
        return pd.Series(np.full(0, np.nan), index=first_series.index)

    if (
        min_data_length > 0
        and first_series is not None
        and len(first_series) < min_data_length
    ):
        return pd.Series(np.full(len(first_series), np.nan), index=first_series.index)

    return None


# =============================================================================
# エラーハンドリングデコレーター
# =============================================================================


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
