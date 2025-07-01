"""
テクニカル指標の共通定数と設定。

オートストラテジーで使用される主要な指標リストと、
その詳細情報、カテゴリ分けを提供します。
"""

from typing import List, Dict, Any

# オートストラテジー用テクニカル指標リスト（10個）
ALL_INDICATORS: List[str] = [
    "SMA",
    "EMA",
    "MACD",
    "BB",
    "RSI",
    "STOCH",
    "CCI",
    "ADX",
    "ATR",
    "OBV",
]

# カテゴリ別指標リスト
TREND_INDICATORS: List[str] = [
    "SMA",
    "EMA",
    "MACD",
    "BB",
]

MOMENTUM_INDICATORS: List[str] = [
    "RSI",
    "STOCH",
    "CCI",
    "ADX",
]

VOLATILITY_INDICATORS: List[str] = [
    "ATR",
]

VOLUME_INDICATORS: List[str] = [
    "OBV",
]

# 現在使用されていないカテゴリ（空リスト）
PRICE_TRANSFORM_INDICATORS: List[str] = []

# 指標情報辞書
INDICATOR_INFO: Dict[str, Dict[str, Any]] = {
    "SMA": {"name": "Simple Moving Average", "category": "trend", "min_period": 2},
    "EMA": {"name": "Exponential Moving Average", "category": "trend", "min_period": 2},
    "MACD": {
        "name": "Moving Average Convergence Divergence",
        "category": "trend",
        "min_period": 2,
    },
    "BB": {"name": "Bollinger Bands", "category": "volatility", "min_period": 2},
    "RSI": {"name": "Relative Strength Index", "category": "momentum", "min_period": 2},
    "STOCH": {"name": "Stochastic", "category": "momentum", "min_period": 2},
    "CCI": {"name": "Commodity Channel Index", "category": "momentum", "min_period": 2},
    "ADX": {
        "name": "Average Directional Movement Index",
        "category": "momentum",
        "min_period": 2,
    },
    "ATR": {"name": "Average True Range", "category": "volatility", "min_period": 2},
    "OBV": {"name": "On Balance Volume", "category": "volume", "min_period": 1},
}


def validate_indicator_lists() -> bool:
    """指標リストの整合性を検証"""
    category_total = (
        len(TREND_INDICATORS)
        + len(MOMENTUM_INDICATORS)
        + len(VOLATILITY_INDICATORS)
        + len(VOLUME_INDICATORS)
        + len(PRICE_TRANSFORM_INDICATORS)
    )

    return len(ALL_INDICATORS) == category_total == len(INDICATOR_INFO)


TOTAL_INDICATORS = len(ALL_INDICATORS)
assert TOTAL_INDICATORS == 9, f"Expected 9 indicators, got {TOTAL_INDICATORS}"
assert validate_indicator_lists(), "Indicator lists are inconsistent"
