"""
列挙型定義

ポジションサイジングやTP/SL方式のEnumを定義します。
"""

from __future__ import annotations

from enum import Enum


class PositionSizingMethod(Enum):
    """ポジションサイジング決定方式"""

    HALF_OPTIMAL_F = "half_optimal_f"
    VOLATILITY_BASED = "volatility_based"
    FIXED_RATIO = "fixed_ratio"
    FIXED_QUANTITY = "fixed_quantity"


class TPSLMethod(Enum):
    """TP/SL決定方式"""

    FIXED_PERCENTAGE = "fixed_percentage"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    VOLATILITY_BASED = "volatility_based"
    STATISTICAL = "statistical"
    ADAPTIVE = "adaptive"
