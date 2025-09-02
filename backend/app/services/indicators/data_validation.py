"""
データ長検証強化モジュール

運用データのNaNリスクを軽減するためのデータ長チェック強化。
各指標の最小必要データ長を定義し、運用データでのデータ長不足時の
警告出力とNaN回避処理を提供。
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

from .config.pandas_ta_config import PANDAS_TA_CONFIG

logger = logging.getLogger(__name__)

# 各指標の最小必要データ長定義
def get_param_value(params, keys, default):
    """パラメータ名がlengthまたはwindowの場合の値取得をサポート"""
    for key in keys:
        if key in params:
            return params[key]
    return default

INDICATOR_MIN_DATA_LENGTHS = {
    "RSI": lambda params: get_param_value(params, ["length", "window"], 14),
    "SMA": lambda params: get_param_value(params, ["length", "window"], 20),
    "EMA": lambda params: max(2, get_param_value(params, ["length", "window"], 20) // 3),  # EMAは初期値のために3分の1程度
    "WMA": lambda params: get_param_value(params, ["length", "window"], 20),
    "MACD": lambda params: params.get("slow", 26) + params.get("signal", 9) + 5,
    "STOCHF": lambda params: params.get("fastk_length", 5) + params.get("fastd_length", 3),
    "SUPERTREND": lambda params: get_param_value(params, ["length", "window"], 10) + 10,
    "BBANDS": lambda params: get_param_value(params, ["length", "window"], 20),
    "AO": lambda params: 34,  # AO needs 34 data points for calculation
    "PPO": lambda params: params.get("slow", 26) + params.get("signal", 9),
    "STOCHRSI": lambda params: get_param_value(params, ["length", "window"], 14) + params.get("k_period", 5) + params.get("d_period", 3),
    "KST": lambda params: max(params.get("roc4", 30) + params.get("n4", 15), params.get("signal", 9) + 10),
    "TEMA": lambda params: max(3, get_param_value(params, ["length", "window"], 14) // 2),
    "ALMA": lambda params: get_param_value(params, ["length", "window"], 9),
    "AROON": lambda params: get_param_value(params, ["length", "window"], 14) + 1,
    "UI": lambda params: get_param_value(params, ["length", "window"], 14),
    "SINWMA": lambda params: get_param_value(params, ["length", "window"], 14),
    "RVI": lambda params: get_param_value(params, ["length", "window"], 14),
    "UO": lambda params: params.get("slow", 28) + 10,
    "FWMA": lambda params: get_param_value(params, ["length", "window"], 10),
}


class DataValidationError(Exception):
    """データ検証エラー"""
    pass


def get_minimum_data_length(indicator_type: str, params: Dict[str, Any]) -> int:
    """
    指標の種類とパラメータから最小必要データ長を取得

    Args:
        indicator_type: 指標タイプ
        params: パラメータ辞書

    Returns:
        最小必要データ長
    """
    if indicator_type in INDICATOR_MIN_DATA_LENGTHS:
        return INDICATOR_MIN_DATA_LENGTHS[indicator_type](params)

    # デフォルト値 - lengthまたはwindowパラメータをサポート
    config = PANDAS_TA_CONFIG.get(indicator_type)
    if config:
        # lengthまたはwindowパラメータを取得
        length_value = get_param_value(config["default_values"], ["length", "window"], 14)
        return length_value

    return 1  # 最低1つのデータ点


def validate_data_length(
    df: pd.DataFrame,
    indicator_type: str,
    params: Dict[str, Any]
) -> bool:
    """
    強化されたデータ長検証

    Args:
        df: OHLCV価格データ
        indicator_type: 指標タイプ
        params: パラメータ辞書

    Returns:
        データ長が十分な場合はTrue
    """
    required_length = get_minimum_data_length(indicator_type, params)
    data_length = len(df)

    if data_length < required_length:
        logger.warning(
            "data_validation.pyを強化したデータ長検証における警告。"
            f"{indicator_type}: 必要なデータ長 {required_length} に対して "
            f"実際の長さ {data_length} が不足しています"
        )
        return False

    return True


def validate_data_length_with_fallback(
    df: pd.DataFrame,
    indicator_type: str,
    params: Dict[str, Any]
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
    absolute_minimum = get_absolute_minimum_length(indicator_type)
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
        "EMA": 2,  # EMA needs at least 2 data points
        "TEMA": 3,  # TEMA needs at least 3
        "MACD": 26 + 9 + 3,  # slow + signal + some buffer
        "STOCHF": 5 + 3 + 3,  # fastk_length + fastd_length + buffer
        "KST": 30 + 15 + 10,  # largest roc + signal + buffer
        "AO": 34,  # AO specific minimum
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


def validate_ohlcv_data_quality(df: pd.DataFrame) -> List[str]:
    """
    OHLCVデータの品質検証（prometheus用）

    Args:
        df: OHLCVデータフレーム

    Returns:
        検出された問題のリスト
    """
    issues = []

    # データ長チェック - 指標計算に最低限必要な長さを考慮
    min_required = 3  # 絶対的最小
    if len(df) < min_required:
        issues.append(f"データ長不足: {len(df)} (最小{min_required}以上必要)")

    recommended_min = 20  # 推奨最小
    if len(df) < recommended_min:
        issues.append(f"データ長推奨不足: {len(df)} (推奨{recommended_min}以上)")

    # NaNチェック
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        for col, count in nan_counts.items():
            if count > 0:
                issues.append(f"NaN検出 {col}: {count}個")

    # ゼロ値チェック
    zero_cols = ['Volume']
    for col in zero_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            if zero_count > len(df) * 0.1:  # 10%以上
                issues.append(f"ゼロ値過多 {col}: {zero_count}個")

    return issues


def validate_indicator_params(indicator_type: str, params: Dict[str, Any]) -> bool:
    """
    パラメータ検証（prometheus用）

    Args:
        indicator_type: 指標タイプ
        params: パラメータ辞書

    Returns:
        パラメータが有効かどうか
    """
    config = PANDAS_TA_CONFIG.get(indicator_type)
    if not config:
        logger.warning(f"未知の指標タイプ: {indicator_type}")
        return False

    for param_name, aliases in config["params"].items():
        for alias in aliases:
            if alias in params:
                value = params[alias]
                if not is_valid_param_value(param_name, value):
                    logger.warning(f"無効なパラメータ値: {indicator_type}.{alias} = {value}")
                    return False
                break

    return True


def is_valid_param_value(param_name: str, value: Any) -> bool:
    """
    パラメータ値の有効性検証

    Args:
        param_name: パラメータ名
        value: 値

    Returns:
        有効かどうか
    """
    if not isinstance(value, (int, float)):
        return False

    if param_name in ["length", "window"] and (value <= 0 or value > 1000):
        return False

    return True