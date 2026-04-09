"""
Auto Strategy 共通定数

システム全体で使用される共通定数を定義します。
"""

from app.config.constants import (  # noqa: F401
    DEFAULT_MARKET_SYMBOL,
    SUPPORTED_TIMEFRAMES,
)

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
    "max_position_size": 9999.0,
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

# === AutoStrategyConfig デフォルト値（環境変数ベース設定用） ===
AUTO_STRATEGY_CONFIG_DEFAULTS = {
    # 戦略生成制約
    "tournament_size": 3,
    "min_indicators": 2,
    "min_conditions": 2,
    "max_conditions": 5,
    # 多目的最適化設定
    "enable_multi_objective": False,
    "objectives": ["total_return"],
    "objective_weights": [1.0],
    # フィットネス共有設定（GA_DEFAULT_FITNESS_SHARINGを上書き）
    "enable_fitness_sharing": False,
    "fitness_sharing_radius": 0.1,
    "sharing_alpha": 1.0,
    # フォールバック設定
    "fallback_symbol": "BTC/USDT:USDT",
    "fallback_timeframe": "1d",
    "fallback_start_date": "2024-01-01",
    "fallback_end_date": "2024-04-09",
    "fallback_initial_capital": 100000.0,
    "fallback_commission_rate": 0.001,
    # 戦略API設定
    "default_strategies_limit": 20,
    "max_strategies_limit": 100,
}
