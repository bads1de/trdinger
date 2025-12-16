"""
Auto Strategy 共通定数とEnum定義
"""

from enum import Enum

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

    MARKET = "market"  # 成行注文（現行デフォルト）
    LIMIT = "limit"  # 指値注文（有利な価格での約定を狙う）
    STOP = "stop"  # 逆指値注文（ブレイクアウト戦略向け）
    STOP_LIMIT = "stop_limit"  # 逆指値指値注文（より精密な制御）


class IndicatorType(str, Enum):
    """指標分類"""

    MOMENTUM = "momentum"  # モメンタム系
    TREND = "trend"  # トレンド系
    VOLATILITY = "volatility"  # ボラティリティ系


# === 演算子定数 ===
OPERATORS = [
    ">",
    "<",
    ">=",
    "<=",
    "==",
    "!=",
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
    "BTC/USDT:USDT",
]

DEFAULT_SYMBOL = "BTC/USDT:USDT"

# === 時間軸定数 ===
SUPPORTED_TIMEFRAMES = [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
]

DEFAULT_TIMEFRAME = "1h"

# 複合指標
COMPOSITE_INDICATORS = [
    "ICHIMOKU",  # 一目均衡表 (Ichimoku Kinko Hyo)
    "UI",  # Ulcer指数 (Ulcer Index)
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
    "stop_loss_pct": (0.005, 0.15),  # 0.5% ~ 15%
    "take_profit_pct": (0.01, 0.3),  # 1% ~ 30%
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

# === GA基本設定 ===
GA_DEFAULT_CONFIG = {
    "population_size": 100,
    "generations": 50,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1,
    "elite_size": 10,
    "max_indicators": 3,
    "zero_trades_penalty": 0.1,
    "constraint_violation_penalty": 0.0,
}

# === フィットネス重み設定 ===
FITNESS_WEIGHT_PROFILES = {
    "balanced": {
        "total_return": 0.2,
        "sharpe_ratio": 0.25,
        "max_drawdown": 0.15,
        "win_rate": 0.1,
        "balance_score": 0.1,
        "ulcer_index_penalty": 0.15,
        "trade_frequency_penalty": 0.05,
    },
}

DEFAULT_FITNESS_WEIGHTS = FITNESS_WEIGHT_PROFILES["balanced"]

# === フィットネス制約設定 ===
DEFAULT_FITNESS_CONSTRAINTS = {
    "min_trades": 10,
    "max_drawdown_limit": 0.3,
    "min_sharpe_ratio": 1.0,
}

# === GA目的設定 ===
DEFAULT_GA_OBJECTIVES = ["weighted_score"]
DEFAULT_GA_OBJECTIVE_WEIGHTS = [1.0]

# === GAフィットネス共有設定 ===
GA_DEFAULT_FITNESS_SHARING = {
    "enable_fitness_sharing": True,
    "sharing_radius": 0.1,
    "sharing_alpha": 1.0,
    "sampling_threshold": 200,
    "sampling_ratio": 0.3,
}

# === GAパラメータ範囲定義 ===
GA_PARAMETER_RANGES = {
    # 基本パラメータ
    "period": [5, 200],
    "fast_period": [5, 20],
    "slow_period": [20, 50],
    "signal_period": [5, 15],
    # 特殊パラメータ
    "std_dev": [1.5, 2.5],
    "k_period": [10, 20],
    "d_period": [3, 7],
    "slowing": [1, 5],
    # 閾値パラメータ
    "overbought": [70, 90],
    "oversold": [10, 30],
}

# === GA閾値範囲定義 ===
GA_THRESHOLD_RANGES = {
    "oscillator_0_100": [20, 80],
    "oscillator_plus_minus_100": [-100, 100],
    "momentum_zero_centered": [-0.5, 0.5],
    "funding_rate": [0.0001, 0.0005, 0.001, -0.0001, -0.0005, -0.001],
    "open_interest": [1000000, 5000000, 10000000, 50000000],
    "price_ratio": [0.95, 1.05],
}

# === GA突然変異設定 ===
GA_MUTATION_SETTINGS = {
    "indicator_param_mutation_range": (0.8, 1.2),
    "indicator_add_delete_probability": 0.3,
    "indicator_add_vs_delete_probability": 0.5,
    "crossover_field_selection_probability": 0.5,
    "condition_operator_switch_probability": 0.5,
    "condition_change_probability_multiplier": 0.5,
    "condition_selection_probability": 0.5,
    "risk_param_mutation_range": (0.8, 1.2),
    "tpsl_gene_creation_probability_multiplier": 0.2,
    "position_sizing_gene_creation_probability_multiplier": 0.2,
    "adaptive_mutation_variance_threshold": 0.1,
    "adaptive_mutation_rate_decrease_multiplier": 0.5,
    "adaptive_mutation_rate_increase_multiplier": 2.0,
    "valid_condition_operators": [">", "<", ">=", "<=", "=="],
    "numeric_threshold_probability": 0.3,
    "min_compatibility_score": 0.5,
    "strict_compatibility_score": 0.7,
}
