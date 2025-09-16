"""
データ長検証強化モジュール

運用データのNaNリスクを軽減するためのデータ長チェック強化。
各指標の最小必要データ長を定義し、運用データでのデータ長不足時の
警告出力とNaN回避処理を提供。
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from .config.indicator_definitions import PANDAS_TA_CONFIG

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
    config = PANDAS_TA_CONFIG.get(indicator_type.upper())
    if config and "min_length" in config:
        if callable(config["min_length"]):
            return config["min_length"](params)
        else:
            return config["min_length"]

    # フォールバック：デフォルト値 - lengthまたはwindowパラメータをサポート
    if config:
        # lengthまたはwindowパラメータを取得
        length_value = get_param_value(
            config["default_values"], ["length", "window"], 14
        )
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
    standard_length = get_minimum_data_length(indicator_type, params)
    data_length = len(df)

    if data_length >= standard_length:
        return True, standard_length

    # 各指標ごとの最小データ長を計算
    absolute_minimum = get_absolute_minimum_length(indicator_type.upper())
    min_required = max(absolute_minimum, standard_length // 3)

    if data_length >= min_required:
        logger.info(
            "データ長不足時は、この関数でフォールバック加工を実行。"
            f"{indicator_type}: 標準データ長 {standard_length} が {data_length} 不足のため、"
            f"最小データ長 {min_required} にフォールしNaNフィルタを適用"
        )
        return True, data_length

    logger.warning(
        "この関数で運用データでの警告出力を強化。"
        f"{indicator_type}: 必要なデータ長 {standard_length} 以上、"
        f"最低データ長 {min_required} が {data_length} 不足のためNaNフィルタを適用"
    )
    return False, min_required


def get_absolute_minimum_length(indicator_type: str) -> int:
    """
    各指標の絶対的最小データ長を取得

    Args:
        indicator_type: 指標タイプ

    Returns:
        絶対的最小データ長
    """
    absolute_mins = {
        "SMA": 2,  # SMA needs at least 2 data points
        "EMA": 2,  # EMA needs at least 2 data points
        "TEMA": 3,  # TEMA needs at least 3
        "MACD": 26 + 9 + 3,  # slow + signal + some buffer
    }

    return absolute_mins.get(indicator_type, 1)


def create_nan_result(df: pd.DataFrame, indicator_type: str) -> np.ndarray:
    """
    データ長不足時のNaN結果生成

    Args:
        df: 元のデータフレーム
        indicator_type: 指標タイプ

    Returns:
        NaN配列
    """
    config = PANDAS_TA_CONFIG.get(indicator_type)
    data_length = len(df)

    if not config:
        return np.full(data_length, np.nan)

    if config["returns"] == "single":
        return np.full(data_length, np.nan)
    else:
        return_cols = config.get("return_cols", ["Result"])
        nan_result = np.full((data_length, len(return_cols)), np.nan)
        return nan_result


