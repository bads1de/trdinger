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


class EntryType(Enum):
    """エントリー注文タイプ"""

    MARKET = "market"  # 成行注文（現行デフォルト）
    LIMIT = "limit"  # 指値注文（有利な価格での約定を狙う）
    STOP = "stop"  # 逆指値注文（ブレイクアウト戦略向け）
    STOP_LIMIT = "stop_limit"  # 逆指値指値注文（より精密な制御）





