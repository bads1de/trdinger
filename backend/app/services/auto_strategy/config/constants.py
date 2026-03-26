"""
Auto Strategy 共通定数とEnum定義

GA固有の定数は ga_constants.py に分離されています。
後方互換性のため、ga_constants.py の定数はここからも再エクスポートされます。
"""

from enum import Enum

from app.config.constants import DEFAULT_MARKET_SYMBOL, SUPPORTED_TIMEFRAMES  # noqa: F401

# GA固有定数のre-export（後方互換性維持）
from .ga_constants import (  # noqa: F401
    DEFAULT_FITNESS_CONSTRAINTS,
    DEFAULT_FITNESS_WEIGHTS,
    DEFAULT_GA_OBJECTIVE_WEIGHTS,
    DEFAULT_GA_OBJECTIVES,
    FITNESS_WEIGHT_PROFILES,
    GA_DEFAULT_CONFIG,
    GA_DEFAULT_FITNESS_SHARING,
    GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS,
    GA_DEFAULT_TPSL_METHOD_CONSTRAINTS,
    GA_MUTATION_SETTINGS,
    GA_PARAMETER_RANGES,
    GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE,
    GA_POSITION_SIZING_ATR_PERIOD_RANGE,
    GA_POSITION_SIZING_FIXED_QUANTITY_RANGE,
    GA_POSITION_SIZING_FIXED_RATIO_RANGE,
    GA_POSITION_SIZING_LOOKBACK_RANGE,
    GA_POSITION_SIZING_MAX_ES_RATIO_RANGE,
    GA_POSITION_SIZING_MAX_SIZE_RANGE,
    GA_POSITION_SIZING_MAX_VAR_RATIO_RANGE,
    GA_POSITION_SIZING_MIN_SIZE_RANGE,
    GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE,
    GA_POSITION_SIZING_PRIORITY_RANGE,
    GA_POSITION_SIZING_RISK_PER_TRADE_RANGE,
    GA_POSITION_SIZING_VAR_CONFIDENCE_RANGE,
    GA_POSITION_SIZING_VAR_LOOKBACK_RANGE,
    GA_TPSL_ATR_MULTIPLIER_RANGE,
    GA_TPSL_RR_RANGE,
    GA_TPSL_SL_RANGE,
    GA_TPSL_TP_RANGE,
    GA_THRESHOLD_RANGES,
    POSITION_SIZING_LIMITS,
)


# === 列挙型定義 ===


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

# === データソース定数 ===
DATA_SOURCES = [
    "close",
    "open",
    "high",
    "low",
    "volume",
    "OpenInterest",
    "FundingRate",
]

# === 取引ペア定数 ===
SUPPORTED_SYMBOLS = [
    DEFAULT_MARKET_SYMBOL,
]

DEFAULT_SYMBOL = DEFAULT_MARKET_SYMBOL

# === 時間軸定数 ===

DEFAULT_TIMEFRAME = "1h"

# 複合指標
COMPOSITE_INDICATORS = [
    "ICHIMOKU",
    "UI",
]


# 移動平均系指標の定数
MOVING_AVERAGE_INDICATORS = {
    "SMA",
    "EMA",
    "WMA",
    "KAMA",
    "TEMA",
    "DEMA",
    "T3",
}

# 優先的な移動平均指標（MA系を2本以上生成する場合の候補）
PREFERRED_MA_INDICATORS = {"SMA", "EMA"}

# periodパラメータが必要な移動平均指標
MA_INDICATORS_NEEDING_PERIOD = {
    "SMA",
    "EMA",
    "WMA",
    "KAMA",
    "TEMA",
    "DEMA",
    "T3",
}

# === TP/SL関連定数 ===
TPSL_METHODS = [m.value for m in TPSLMethod]

TPSL_LIMITS = {
    "stop_loss_pct": (0.005, 0.15),
    "take_profit_pct": (0.01, 0.3),
    "base_stop_loss": (0.005, 0.15),
    "atr_multiplier_sl": (0.5, 5.0),
    "atr_multiplier_tp": (1.0, 10.0),
    "atr_period": (5, 50),
    "lookback_period": (20, 500),
    "confidence_threshold": (0.1, 1.0),
}

# === ポジションサイジング関連定数 ===
POSITION_SIZING_METHODS = [m.value for m in PositionSizingMethod]


# === エラーコード定数 ===
ERROR_CODES = {
    "GA_ERROR": "GA処理エラー",
    "STRATEGY_GENERATION_ERROR": "戦略生成エラー",
    "CALCULATION_ERROR": "計算エラー",
    "VALIDATION_ERROR": "検証エラー",
    "DATA_ERROR": "データエラー",
    "CONFIG_ERROR": "設定エラー",
}

# === 取引制約定数 ===
CONSTRAINTS = {
    "min_trades": 10,
    "max_drawdown_limit": 0.3,
    "max_position_size": 1.0,
    "min_position_size": 0.01,
}

# === 自動戦略デフォルト値 (フォールバック用) ===
AUTO_STRATEGY_DEFAULTS = {
    # ボラティリティTP/SL
    "atr_period": 14,
    "atr_multiplier_sl": 2.0,
    "atr_multiplier_tp": 3.0,
    "min_sl_pct": 0.005,
    "max_sl_pct": 0.1,
    "min_tp_pct": 0.01,
    "max_tp_pct": 0.2,
    # ポジションサイジング
    "default_atr_multiplier": 0.02,
    "fallback_atr_multiplier": 0.04,
    "assumed_win_rate": 0.55,
    "assumed_avg_win": 0.02,
    "assumed_avg_loss": 0.015,
    "default_position_ratio": 0.1,
}
