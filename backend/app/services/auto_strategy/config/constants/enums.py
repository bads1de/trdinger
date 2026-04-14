"""
Auto Strategy Enum定義

列挙型定義を提供します。
"""

from enum import Enum


class PositionSizingMethod(str, Enum):
    """ポジションサイジング決定方式"""

    HALF_OPTIMAL_F = "half_optimal_f"
    VOLATILITY_BASED = "volatility_based"
    FIXED_RATIO = "fixed_ratio"
    FIXED_QUANTITY = "fixed_quantity"


class TPSLMethod(str, Enum):
    """TP/SL決定方式"""

    FIXED_PERCENTAGE = "fixed_percentage"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    VOLATILITY_BASED = "volatility_based"
    STATISTICAL = "statistical"
    ADAPTIVE = "adaptive"


class EntryType(str, Enum):
    """エントリー注文タイプ"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class ExitType(str, Enum):
    """イグジット注文タイプ"""

    FULL = "full"  # 全ポジション決済
    PARTIAL = "partial"  # 部分決済
    TRAILING = "trailing"  # トレーリングSL起動（決済しない）


class IndicatorType(str, Enum):
    """指標分類"""

    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"


# === 演算子定数 ===
OPERATORS = [
    ">",
    "<",
    ">=",
    "<=",
    "==",
    "!=",
    "CROSS_UP",
    "CROSS_DOWN",
]
