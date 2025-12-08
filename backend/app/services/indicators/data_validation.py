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


# 各指標の絶対的最小データ長定義
# この値は、指標が意味のある結果を返すために必要な最低限のデータポイント数
# パラメータに関係なく必要な固定値
_ABSOLUTE_MINIMUM_LENGTHS = {
    # トレンド系
    "SMA": 2,  # 最低2点で平均を計算可能
    "EMA": 2,  # 最低2点で指数平滑を計算可能
    "DEMA": 3,  # 2重指数平滑のため最低3点
    "TEMA": 3,  # 三重指数平滑のため最低3点
    "WMA": 2,  # 加重移動平均
    "HMA": 4,  # Hull移動平均（内部でWMAを使用）
    "KAMA": 3,  # Kaufman適応移動平均
    "ZLMA": 3,  # ゼロラグ移動平均
    "LINREG": 3,  # 線形回帰（最低3点で傾き計算可能）
    # モメンタム系
    "RSI": 3,  # 最低3点で上昇/下降を判定可能
    "MACD": 38,  # slow(26) + signal(9) + buffer(3)
    "STOCH": 5,  # ストキャスティクス
    "CCI": 3,  # 商品チャネル指数
    "ROC": 2,  # 変化率
    "MOM": 2,  # モメンタム
    "WILLR": 3,  # Williams %R
    "TSI": 5,  # True Strength Index
    "CMO": 3,  # チャンデモメンタム
    # ボラティリティ系
    "ATR": 3,  # 平均真の値幅
    "NATR": 3,  # 正規化ATR
    "BBANDS": 3,  # ボリンジャーバンド
    "KELTNER": 5,  # ケルトナーチャネル
    "DONCHIAN": 2,  # ドンチャンチャネル
    "SUPERTREND": 5,  # スーパートレンド
    # 出来高系
    "OBV": 2,  # オンバランスボリューム
    "AD": 2,  # A/Dライン
    "MFI": 3,  # マネーフローインデックス
    "VWAP": 2,  # 出来高加重平均価格
}


def get_absolute_minimum_length(indicator_type: str) -> int:
    """
    各指標の絶対的最小データ長を取得

    この関数は、パラメータに関係なく指標が正常に動作するために
    必要な最低限のデータポイント数を返します。

    Args:
        indicator_type: 指標タイプ

    Returns:
        絶対的最小データ長（デフォルト: 1）
    """
    return _ABSOLUTE_MINIMUM_LENGTHS.get(indicator_type, 1)


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
