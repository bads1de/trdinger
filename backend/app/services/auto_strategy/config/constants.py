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

# === GA基本設定 ===
GA_DEFAULT_CONFIG = {
    "population_size": 100,
    "generations": 50,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1,
    "elite_size": 10,
    # バリデーション時のデフォルト上限として使用されるため、比較的緩めに設定
    # 実際のGA実行時はフロントエンドからの設定が優先される
    "max_indicators": 10,
    "zero_trades_penalty": 0.1,
    "constraint_violation_penalty": 0.0,
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
    "valid_condition_operators": [
        ">",
        "<",
        ">=",
        "<=",
        "==",
        "CROSS_UP",
        "CROSS_DOWN",
    ],
    "numeric_threshold_probability": 0.3,
    "min_compatibility_score": 0.5,
    "strict_compatibility_score": 0.7,
}

# === GA TPSL関連定数 ===
GA_DEFAULT_TPSL_METHOD_CONSTRAINTS = [
    "fixed_percentage",
    "risk_reward_ratio",
    "volatility_based",
    "statistical",
    "adaptive",
]

GA_TPSL_SL_RANGE = [0.01, 0.08]  # SL範囲（1%-8%）
GA_TPSL_TP_RANGE = [0.02, 0.20]  # TP範囲（2%-20%）
GA_TPSL_RR_RANGE = [1.2, 4.0]  # リスクリワード比範囲
GA_TPSL_ATR_MULTIPLIER_RANGE = [1.0, 4.0]  # ATR倍率範囲

# === GA ポジションサイジング関連定数 ===
GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS = [
    "half_optimal_f",
    "volatility_based",
    "fixed_ratio",
    "fixed_quantity",
]

GA_POSITION_SIZING_LOOKBACK_RANGE = [50, 200]  # ハーフオプティマルF用ルックバック期間
GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE = [0.25, 0.75]  # オプティマルF倍率範囲
GA_POSITION_SIZING_ATR_PERIOD_RANGE = [10, 30]  # ATR計算期間範囲
GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE = [
    1.0,
    4.0,
]  # ポジションサイジング用ATR倍率範囲
GA_POSITION_SIZING_RISK_PER_TRADE_RANGE = [
    0.01,
    0.05,
]  # 1取引あたりのリスク範囲（1%-5%）
GA_POSITION_SIZING_FIXED_RATIO_RANGE = [0.05, 0.3]  # 固定比率範囲（5%-30%）
GA_POSITION_SIZING_FIXED_QUANTITY_RANGE = [0.1, 5.0]  # 固定枚数範囲
GA_POSITION_SIZING_MIN_SIZE_RANGE = [0.01, 0.1]  # 最小ポジションサイズ範囲
GA_POSITION_SIZING_MAX_SIZE_RANGE = [
    0.001,
    1.0,
]  # 最大ポジションサイズ範囲（システム全体のmax_position_sizeに一致）
GA_POSITION_SIZING_PRIORITY_RANGE = [0.5, 1.5]  # 優先度範囲
GA_POSITION_SIZING_VAR_CONFIDENCE_RANGE = [0.8, 0.99]  # VaR信頼水準
GA_POSITION_SIZING_MAX_VAR_RATIO_RANGE = [0.005, 0.05]  # VaR許容比率
GA_POSITION_SIZING_MAX_ES_RATIO_RANGE = [0.01, 0.1]  # ES許容比率
GA_POSITION_SIZING_VAR_LOOKBACK_RANGE = [50, 500]  # VaR計算のルックバック期間

# ポジションサイジング制限設定
POSITION_SIZING_LIMITS = {
    "lookback_period": (10, 500),
    "optimal_f_multiplier": (0.1, 1.0),
    "atr_period": (5, 50),
    "atr_multiplier": (0.5, 10.0),
    "risk_per_trade": (0.001, 0.1),
    "fixed_ratio": (0.01, 10.0),
    "fixed_quantity": (0.01, 1000.0),
    "min_position_size": (0.001, 1.0),
    "max_position_size": (0.001, 1.0),
    "var_confidence": (0.8, 0.999),
    "max_var_ratio": (0.001, 0.1),
    "max_expected_shortfall_ratio": (0.001, 0.2),
    "var_lookback": (20, 1000),
}
