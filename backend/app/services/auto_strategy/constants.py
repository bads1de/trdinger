"""
Auto Strategy 共通定数

"""

from typing import List
from enum import Enum


class IndicatorType(str, Enum):
    """指標分類"""

    MOMENTUM = "momentum"  # モメンタム系
    TREND = "trend"  # トレンド系
    VOLATILITY = "volatility"  # ボラティリティ系


# 戦略タイプ定義
class StrategyType(str, Enum):
    """戦略タイプ"""

    DIFFERENT_INDICATORS = "different_indicators"  # 異なる指標の組み合わせ
    TIME_SEPARATION = "time_separation"  # 時間軸分離
    COMPLEX_CONDITIONS = "complex_conditions"  # 複合条件
    INDICATOR_CHARACTERISTICS = "indicator_characteristics"  # 指標特性活用


# === 演算子定数 ===
OPERATORS = [
    ">",
    "<",
    ">=",
    "<=",
    "==",
    "!=",
    "above",
    "below",
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

# 全テクニカル指標（indicator_registryに登録されているもの）-- utils/indicator_utils.py に移行済み
# VALID_INDICATOR_TYPES は get_valid_indicator_types() 関数で動的に取得
VALID_INDICATOR_TYPES: List[str] = (
    []
)  # 後方互換性のための空リスト（使用時は get_valid_indicator_types() を推奨）

# テクニカルオンリー時のおすすめ指標セット（成立性が高い指標を厳選）
CURATED_TECHNICAL_INDICATORS = {
    # モメンタム系（オシレーター）
    "MACD",
    "MACDFIX",
    "MACDEXT",
    "RSI",
    "STOCH",
    "STOCHRSI",
    "CCI",
    "ADX",
    "MFI",
    "WILLR",
    "AROON",
    "AROONOSC",
    "BOP",
    "MOM",
    "ROC",
    "TRIX",
    "TSI",
    "ULTOSC",
    "CMO",
    "DX",
    "MINUS_DI",
    "PLUS_DI",
    "CFO",
    "CHOP",
    "CTI",
    "DPO",
    "RMI",
    "RVI",
    "RVGI",
    "SMI",
    "STC",
    "QQE",
    "VORTEX",
    "PVO",
    # トレンド系（移動平均等）
    "SMA",
    "EMA",
    "WMA",
    "TRIMA",
    "CWMA",
    "KAMA",
    "TEMA",
    "DEMA",
    "ALMA",
    "T3",
    "HMA",
    "RMA",
    "SWMA",
    "ZLMA",
    "VWMA",
    "FWMA",
    "HWMA",
    "JMA",
    "MCGD",
    "VIDYA",
    "LINREG",
    "LINREG_SLOPE",
    "LINREG_INTERCEPT",
    "LINREG_ANGLE",
    "SAR",
    # ボラティリティ系
    "ATR",
    "BBANDS",
    "DONCHIAN",
    "KELTNER",
    "SUPERTREND",
    "NATR",
    "TRANGE",
    "ACCBANDS",
    "HWC",
    "PDIST",
}

# 移動平均系指標の定数
MOVING_AVERAGE_INDICATORS = {
    "SMA",
    "EMA",
    "WMA",
    "TRIMA",
    "KAMA",
    "TEMA",
    "DEMA",
    "ALMA",
    "T3",
    "HMA",
    "RMA",
    "SWMA",
    "ZLMA",
    "MA",
    "VWMA",
    "FWMA",
    "HWMA",
    "JMA",
    "MCGD",
    "VIDYA",
    "WCP",
}

# 優先的な移動平均指標（MA系を2本以上生成する場合の候補）
PREFERRED_MA_INDICATORS = {"SMA", "EMA", "MA", "HMA", "ALMA", "VIDYA", "JMA"}

# periodパラメータが必要な移動平均指標
MA_INDICATORS_NEEDING_PERIOD = {
    "SMA",
    "EMA",
    "WMA",
    "TRIMA",
    "KAMA",
    "TEMA",
    "DEMA",
    "ALMA",
    "T3",
    "HMA",
    "RMA",
    "SWMA",
    "CWMA",
    "ZLMA",
    "VWMA",
    "FWMA",
    "HWMA",
    "JMA",
    "MCGD",
    "VIDYA",
    "WCP",
}

# === ML指標定数 ===
ML_INDICATOR_TYPES = [
    "ML_UP_PROB",  # 機械学習上昇確率 (Machine Learning Up Probability)
    "ML_DOWN_PROB",  # 機械学習下落確率 (Machine Learning Down Probability)
    "ML_RANGE_PROB",  # 機械学習レンジ確率 (Machine Learning Range Probability)
]

# === TP/SL関連定数 ===
TPSL_METHODS = [
    "fixed_percentage",
    "risk_reward_ratio",
    "volatility_based",
    "statistical",
    "adaptive",
]

# === ポジションサイジング関連定数 ===
POSITION_SIZING_METHODS = [
    "half_optimal_f",
    "volatility_based",
    "fixed_ratio",
    "fixed_quantity",
]

# === バックテスト関連定数 ===
BACKTEST_OBJECTIVES = [
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "sortino_ratio",
]

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
THRESHOLD_RANGES = {
    "oscillator_0_100": [20, 80],
    "oscillator_plus_minus_100": [-100, 100],
    "momentum_zero_centered": [-0.5, 0.5],
    "funding_rate": [0.0001, 0.0005, 0.001, -0.0001, -0.0005, -0.001],
    "open_interest": [1000000, 5000000, 10000000, 50000000],
    "price_ratio": [0.95, 1.05],
}
