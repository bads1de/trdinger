"""
データ長検証強化モジュール

運用データのNaNリスクを軽減するためのデータ長チェック強化。
各指標の最小必要データ長を定義し、運用データでのデータ長不足時の
警告出力とNaN回避処理を提供。
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .config.indicator_config import indicator_registry

logger = logging.getLogger(__name__)


# 各指標の最小必要データ長定義
def get_param_value(params, keys, default):
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
    config = indicator_registry.get_indicator_config(indicator_type.upper())
    if config and config.min_length_func:
        return config.min_length_func(params)

    # フォールバック：デフォルト値 - lengthまたはwindowパラメータをサポート
    if config and config.default_values:
        # lengthまたはwindowパラメータを取得
        length_value = get_param_value(config.default_values, ["length", "window"], 14)
        return length_value

    return 1  # 最低1つのデータ点


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
    # absolute_minimum は絶対に下回れない
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


def get_absolute_minimum_length(indicator_type: str) -> int:
    """
    各指標の絶対的最小データ長を取得
    """
    config = indicator_registry.get_indicator_config(indicator_type.upper())
    if config and hasattr(config, "absolute_min_length") and config.absolute_min_length is not None:
        return int(config.absolute_min_length)
    return 1


def create_nan_result(df: pd.DataFrame, indicator_type: str) -> np.ndarray:
    """
    データ長不足時のNaN結果生成

    Args:
        df: 元のデータフレーム
        indicator_type: 指標タイプ

    Returns:
        NaN配列
    """
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



