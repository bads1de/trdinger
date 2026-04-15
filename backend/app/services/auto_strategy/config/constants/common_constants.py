"""
Auto Strategy 共通定数

システム全体で使用される共通定数を定義します。
"""

from app.config.constants import DEFAULT_MARKET_SYMBOL

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
DEFAULT_SYMBOL = DEFAULT_MARKET_SYMBOL

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
TPSL_METHODS = ["fixed_percentage", "risk_reward_ratio", "volatility_based", "statistical", "adaptive"]

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
POSITION_SIZING_METHODS = ["half_optimal_f", "volatility_based", "fixed_ratio", "fixed_quantity"]

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

# === 戦略一覧APIとGAフォールバックの既定値 ===
DEFAULT_STRATEGIES_LIMIT = 20
MAX_STRATEGIES_LIMIT = 100
GA_FALLBACK_START_DATE = "2024-01-01"
GA_FALLBACK_END_DATE = "2024-04-09"
