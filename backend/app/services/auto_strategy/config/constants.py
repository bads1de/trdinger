"""
Auto Strategy 共通定数

"""

from typing import Dict, List

# === 指標タイプ定義 ===
from enum import Enum

class IndicatorType(str, Enum):
    """指標分類"""
    MOMENTUM = "momentum"  # モメンタム系
    TREND = "trend"  # トレンド系
    VOLATILITY = "volatility"  # ボラティリティ系
    STATISTICS = "statistics"  # 統計系
    PATTERN_RECOGNITION = "pattern_recognition"  # パターン認識系

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

# === テクニカル指標定数 ===
# indicator_registryに登録されているすべての指標を含む完全なリスト
VALID_INDICATOR_TYPES = [
    # 数学変換系指標
    "ACOS",
    "ASIN",
    "ATAN",
    "CEIL",
    "COS",
    "COSH",
    "EXP",
    "FLOOR",
    "LN",
    "LOG10",
    "SIN",
    "SINH",
    "SQRT",
    "TAN",
    "TANH",
    # ボリューム系指標
    "AD",
    "ADOSC",
    "OBV",
    "EOM",
    "KVO",
    "CMF",
    "NVI",
    "PVI",
    "PVT",
    "VWAP",
    "VP",    # Volume Price Confirmation
    "PVOL",  # Price-Volume indicator
    "PVR",   # Price Volume Rank indicator
    "EFI",   # Elder's Force Index
    "AOBV",  # Archer On-Balance Volume
    # 数学演算子
    "ADD",
    "DIV",
    "MULT",
    "SUB",
    "MAX",
    "MIN",
    "SUM",
    # モメンタム系指標
    "ADX",
    "ADXR",
    "AO",
    "APO",
    "AROON",
    "AROONOSC",
    "BOP",
    "CCI",
    "CFO",
    "CHOP",
    "CTI",
    "DPO",
    "DX",
    "MFI",
    "MINUS_DI",
    "MINUS_DM",
    "PLUS_DI",
    "PLUS_DM",
    "PPO",
    "QQE",
    "RMI",
    "ROC",
    "ROCP",
    "ROCR",
    "ROCR100",
    "RSI",
    "RSI_EMA_CROSS",
    "RVGI",
    "RVI",
    "SMI",
    "STC",
    "STOCH",
    "STOCHF",
    "STOCHRSI",
    "TRIX",
    "TSI",
    "ULTOSC",
    "VORTEX",
    "WILLR",
    "MACD",
    "MACDEXT",
    "MACDFIX",
    "KDJ",
    "KST",
    "PVO",
    # トレンド系指標
    "SMA",
    "EMA",
    "WMA",
    "TRIMA",
    "KAMA",
    "ALMA",
    "HMA",
    "RMA",
    "SWMA",
    "ZLMA",
    "MA",
    "MIDPOINT",
    "MIDPRICE",
    "SAR",
    "HT_TRENDLINE",
    "PRICE_EMA_RATIO",
    "SMA_SLOPE",
    "VWMA",
    "FWMA",  # Fibonacci's Weighted Moving Average
    "HILO",  # Gann High-Low Activator
    "HL2",   # High-Low Average
    "HLC3",  # High-Low-Close Average
    "HWMA",  # Holt-Winter Moving Average
    "JMA",   # Jurik Moving Average
    "MCGD",  # McGinley Dynamic
    "OHLC4", # Open-High-Low-Close Average
    "PWMA",  # Pascal's Weighted Moving Average
    "SINWMA", # Sine Weighted Moving Average
    "SSF",   # Ehler's Super Smoother Filter
    "VIDYA", # Variable Index Dynamic Average
    "WCP",   # Weighted Closing Price
    # ボラティリティ系指標
    "ATR",
    "NATR",
    "TRANGE",
    "BB",
    "DONCHIAN",
    "KELTNER",
    "SUPERTREND",
    "ABERRATION",  # Aberration
    "ACCBANDS",    # Acceleration Bands
    "HWC",         # Hull-Wilder Channels
    "MASSI",       # Mass Index
    "PDIST",       # Price Distance
    "THERMO",      # Thermometer
    # 統計系指標
    "BETA",
    "CORREL",
    "LINEARREG",
    "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT",
    "LINEARREG_SLOPE",
    "STDDEV",
    "TSF",
    "VAR",
    "ZSCORE",      # Z-Score
    "ENTROPY",     # Entropy
    "KURTOSIS",    # Kurtosis
    "MAD",         # Mean Absolute Deviation
    "MEDIAN",      # Median
    "QUANTILE",    # Quantile
    "SKEW",        # Skewness
    "TOS_STDEVALL", # Standard Deviation All
    "MAXINDEX",    # Max Index
    "MININDEX",    # Min Index
    "MINMAX",      # Min Max
    "MINMAXINDEX", # Min Max Index
    # パターン認識系指標
    "CDL_DOJI",
    "CDL_HAMMER",
    "CDL_HANGING_MAN",
    "CDL_SHOOTING_STAR",
    "CDL_ENGULFING",
    "CDL_HARAMI",
    "CDL_PIERCING",
    "CDL_THREE_BLACK_CROWS",
    "CDL_THREE_WHITE_SOLDIERS",
    "CDL_DARK_CLOUD_COVER",
    "CDL_SPINNING_TOP",
    "CDL_MARUBOZU",
    "CDL_MORNING_STAR",
    "CDL_EVENING_STAR",
    "HAMMER",
    "ENGULFING_PATTERN",
    "MORNING_STAR",
    "EVENING_STAR",
    # Hilbert Transform系指標
    "HT_DCPERIOD",
    "HT_DCPHASE",
    "HT_PHASOR",
    "HT_SINE",
    "HT_TRENDMODE",
    # 複合指標
    "ICHIMOKU",
    # UI indicator (Volatility)
    "UI",
    # 従来の指標（互換性維持）
    "BBANDS",  # BBの別名
    "CMO",  # 未実装だが互換性維持
    "DEMA",  # 未実装だが互換性維持
    "TEMA",  # 未実装だが互換性維持
    "MAMA",  # 未実装だが互換性維持
    "T3",  # 未実装だが互換性維持
    "UO",  # 未実装だが互換性維持
    "MOM",  # 未実装だが互換性維持
]

# === ML指標定数 ===
ML_INDICATOR_TYPES = [
    "ML_UP_PROB",
    "ML_DOWN_PROB",
    "ML_RANGE_PROB",
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

# === GA関連定数 ===
GA_DEFAULT_SETTINGS = {
    "population_size": 10,
    "generations": 5,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1,
    "elite_size": 2,
    "max_indicators": 3,
    "min_indicators": 1,
    "min_conditions": 1,
    "max_conditions": 3,
}

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

# === 閾値設定 ===
THRESHOLD_RANGES = {
    "oscillator_0_100": [20, 80],
    "oscillator_plus_minus_100": [-100, 100],
    "momentum_zero_centered": [-0.5, 0.5],
    "funding_rate": [0.0001, 0.0005, 0.001, -0.0001, -0.0005, -0.001],
    "open_interest": [1000000, 5000000, 10000000, 50000000],
    "price_ratio": [0.95, 1.05],
}

# === 制約設定 ===
CONSTRAINTS = {
    "min_trades": 10,
    "max_drawdown_limit": 0.3,
    "min_sharpe_ratio": 1.0,
    "max_position_size": 1.0,
    "min_position_size": 0.01,
}

# === フィットネス重み設定 ===
FITNESS_WEIGHT_PROFILES = {
    "conservative": {
        "total_return": 0.15,
        "sharpe_ratio": 0.4,
        "max_drawdown": 0.25,
        "win_rate": 0.15,
        "balance_score": 0.05,
    },
    "balanced": {
        "total_return": 0.25,
        "sharpe_ratio": 0.35,
        "max_drawdown": 0.2,
        "win_rate": 0.1,
        "balance_score": 0.1,
    },
    "aggressive": {
        "total_return": 0.4,
        "sharpe_ratio": 0.25,
        "max_drawdown": 0.15,
        "win_rate": 0.1,
        "balance_score": 0.1,
    },
}

# === ユーティリティ関数 ===


def get_all_indicators() -> List[str]:
    """全指標タイプを取得"""
    return VALID_INDICATOR_TYPES + ML_INDICATOR_TYPES


def validate_symbol(symbol: str) -> bool:
    """シンボルの妥当性を検証"""
    return symbol in SUPPORTED_SYMBOLS


def validate_timeframe(timeframe: str) -> bool:
    """時間軸の妥当性を検証"""
    return timeframe in SUPPORTED_TIMEFRAMES


def get_all_indicator_ids() -> Dict[str, int]:
    """
    全指標のIDマッピングを取得（統合版）

    テクニカル指標とML指標を統合したIDマッピングを提供します。
    gene_utils.py との重複機能を統合しています。
    """
    try:
        from app.services.indicators import TechnicalIndicatorService

        indicator_service = TechnicalIndicatorService()
        technical_indicators = list(indicator_service.get_supported_indicators().keys())

        # 全指標を結合
        all_indicators = technical_indicators + ML_INDICATOR_TYPES

        # IDマッピングを作成（空文字列は0、その他は1から開始）
        return {"": 0, **{ind: i + 1 for i, ind in enumerate(all_indicators)}}
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"指標ID取得エラー: {e}")
        return {"": 0}


def get_id_to_indicator_mapping(indicator_ids: Dict[str, int]) -> Dict[int, str]:
    """ID→指標の逆引きマッピングを取得"""
    return {v: k for k, v in indicator_ids.items()}


# === インジケータ解決支援定数 ===
# 複数出力インジケータのデフォルト解決マッピング（例: MACD -> MACD_0）
MULTI_OUTPUT_DEFAULT_MAPPING: Dict[str, str] = {
    "AROON": "AROON_0",
    "MACD": "MACD_0",
    "STOCH": "STOCH_0",
    "BBANDS": "BBANDS_1",  # Middle をデフォルト
}

# 基本的な移動平均インジケータ名のリスト（存在チェックなどに使用）
BASIC_MA_INDICATORS: List[str] = [
    "SMA",
    "EMA",
    "WMA",
    "TRIMA",
    "KAMA",
    "T3",
    "ALMA",
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
]


# === TPSL/Position Sizing 範囲・既定値（散逸防止用） ===
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

POSITION_SIZING_LIMITS = {
    "lookback_period": (10, 500),
    "optimal_f_multiplier": (0.1, 1.0),
    "atr_period": (5, 50),
    "atr_multiplier": (0.5, 10.0),
    "risk_per_trade": (0.001, 0.1),
    "fixed_ratio": (0.01, 10.0),
    "fixed_quantity": (0.01, 1000.0),
    "min_position_size": (0.001, 1.0),
    "max_position_size": (0.001, 1.0),  # GA_MAX_SIZE_RANGEに一致
}

# === GA Config デフォルト定数（一元管理用） ===

# GA基本設定
GA_DEFAULT_CONFIG = {
    "population_size": 100,  # より実用的、デフォルトをGA_DEFAULT_SETTINGSに合わせる
    "generations": 50,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1,
    "elite_size": 10,
    "max_indicators": 3,
}

# フィットネス設定
DEFAULT_FITNESS_WEIGHTS = {
    "total_return": 0.25,
    "sharpe_ratio": 0.35,
    "max_drawdown": 0.2,
    "win_rate": 0.1,
    "balance_score": 0.1,
}

DEFAULT_FITNESS_CONSTRAINTS = {
    "min_trades": 10,
    "max_drawdown_limit": 0.3,
    "min_sharpe_ratio": 1.0,
}

# === 技術指標特性データベース ===
# condition_generator.pyから移行した指標特性定義
INDICATOR_CHARACTERISTICS = {
    "RSI": {
        "type": "momentum",
        "range": (0, 100),
        "long_zones": [(0, 30), (40, 60)],
        "short_zones": [(40, 60), (70, 100)],
        "neutral_zone": (40, 60),
        "oversold_threshold": 30,
        "overbought_threshold": 70,
    },
    "STOCH": {
        "type": "momentum",
        "range": (0, 100),
        "long_zones": [(0, 20), (40, 60)],
        "short_zones": [(40, 60), (80, 100)],
        "neutral_zone": (40, 60),
        "oversold_threshold": 20,
        "overbought_threshold": 80,
    },
    "CCI": {
        "type": "momentum",
        "range": (-200, 200),
        "long_zones": [(-200, -100), (-50, 50)],
        "short_zones": [(-50, 50), (100, 200)],
        "neutral_zone": (-50, 50),
        "oversold_threshold": -100,
        "overbought_threshold": 100,
    },
    "MACD": {
        "type": "momentum",
        "range": None,  # 価格依存
        "zero_cross": True,
        "signal_line": True,
    },
    "MACDEXT": {
        "type": "momentum",
        "range": None,  # 価格依存
        "zero_cross": True,
        "signal_line": True,
    },
    "SMA": {
        "type": "trend",
        "price_comparison": True,
        "trend_following": True,
    },
    "EMA": {
        "type": "trend",
        "price_comparison": True,
        "trend_following": True,
    },
    "MAMA": {"type": "trend", "price_comparison": True, "adaptive": True},
    "ADX": {
        "type": "trend",
        "range": (0, 100),
        "trend_strength": True,
        "no_direction": True,  # 方向性を示さない
        "strong_trend_threshold": 25,
    },
    "BB": {
        "type": "volatility",
        "components": ["upper", "middle", "lower"],
        "mean_reversion": True,
        "breakout_strategy": True,
    },
    "BBANDS": {
        "type": "volatility",
        "components": ["upper", "middle", "lower"],
        "mean_reversion": True,
        "breakout_strategy": True,
    },
    "ATR": {
        "type": "volatility",
        "range": (0, None),
        "volatility_measure": True,
    },
    "BETA": {
        "type": "statistics",
        "range": (-2, 2),  # 一般的なベータ値範囲
        "correlation_measure": True,
        "zero_cross": True,
    },
    "CORREL": {
        "type": "statistics",
        "range": (-1, 1),  # 相関係数
        "correlation_measure": True,
        "zero_cross": True,
    },
    "LINEARREG": {
        "type": "statistics",
        "range": None,  # 価格依存
        "price_comparison": True,
        "trend_following": True,
    },
    "LINEARREG_ANGLE": {
        "type": "statistics",
        "range": (-90, 90),  # 角度
        "zero_cross": True,
        "trend_strength": True,
    },
    "LINEARREG_INTERCEPT": {
        "type": "statistics",
        "range": None,  # 価格依存
        "price_comparison": True,
    },
    "LINEARREG_SLOPE": {
        "type": "statistics",
        "range": None,  # 価格変化率依存
        "zero_cross": True,
        "trend_strength": True,
    },
    "STDDEV": {
        "type": "statistics",
        "range": (0, None),  # 常に正値
        "volatility_measure": True,
    },
    "TSF": {
        "type": "statistics",
        "range": None,  # 価格依存
        "price_comparison": True,
        "trend_following": True,
    },
    "VAR": {
        "type": "statistics",
        "range": (0, None),  # 常に正値
        "volatility_measure": True,
    },
    "CDL_DOJI": {
        "type": "pattern_recognition",
        "range": (-100, 100),  # パターン強度
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
    },
    "ML_UP_PROB": {
        "type": "pattern_recognition",
        "range": (0, 1),  # 確率値
        "ml_prediction": True,
        "long_zones": [(0.6, 1.0)],
        "short_zones": [(0, 0.4)],
        "neutral_zone": (0.4, 0.6),
        "high_confidence_threshold": 0.7,
    },
    "ML_DOWN_PROB": {
        "type": "pattern_recognition",
        "range": (0, 1),  # 確率値
        "ml_prediction": True,
        "long_zones": [(0, 0.4)],
        "short_zones": [(0.6, 1.0)],
        "neutral_zone": (0.4, 0.6),
        "high_confidence_threshold": 0.7,
    },
    "ML_RANGE_PROB": {
        "type": "pattern_recognition",
        "range": (0, 1),  # 確率値
        "ml_prediction": True,
        "long_zones": [(0, 0.3)],
        "short_zones": [(0, 0.3)],
        "neutral_zone": (0.7, 1.0),
        "high_confidence_threshold": 0.8,
    },
    "CDL_HAMMER": {
        "type": "pattern_recognition",
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
        "bullish_pattern": True,
    },
    "CDL_HANGING_MAN": {
        "type": "pattern_recognition",
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
        "bearish_pattern": True,
    },
    "CDL_SHOOTING_STAR": {
        "type": "pattern_recognition",
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
        "bearish_pattern": True,
    },
    "CDL_ENGULFING": {
        "type": "pattern_recognition",
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
    },
    "CDL_HARAMI": {
        "type": "pattern_recognition",
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
    },
    "CDL_PIERCING": {
        "type": "pattern_recognition",
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
        "bullish_pattern": True,
    },
    "CDL_THREE_BLACK_CROWS": {
        "type": "pattern_recognition",
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "continuation_pattern": True,
        "bearish_pattern": True,
    },
    "CDL_THREE_WHITE_SOLDIERS": {
        "type": "pattern_recognition",
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "continuation_pattern": True,
        "bullish_pattern": True,
    },
    "CDL_DARK_CLOUD_COVER": {
        "type": "pattern_recognition",
        "range": (-100, 100),
        "pattern_recognition": True,
        "binary_like": True,
        "reversal_pattern": True,
        "bearish_pattern": True,
    },
    "FWMA": {
        "type": "trend",
        "price_comparison": True,
        "trend_following": True,
        "weighted_average": True,  # フィボナッチ重み付き移動平均
    },
    "SWMA": {
        "type": "trend",
        "price_comparison": True,
        "trend_following": True,
        "symmetric_weighting": True,  # 対称重み付き移動平均
    },
    "VIDYA": {
        "type": "trend",
        "price_comparison": True,
        "trend_following": True,
        "adaptive_smoothing": True,  # 適応的スムージング
        "volatility_adaptive": True,  # ボラティリティ適応型
    },
}

# GA目的設定
DEFAULT_GA_OBJECTIVES = ["total_return"]
DEFAULT_GA_OBJECTIVE_WEIGHTS = [1.0]  # 最大化

# パラメータ範囲定義
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

GA_THRESHOLD_RANGES = {
    "oscillator_0_100": [20, 80],
    "oscillator_plus_minus_100": [-100, 100],
    "momentum_zero_centered": [-0.5, 0.5],
    "funding_rate": [0.0001, 0.0005, 0.001, -0.0001, -0.0005, -0.001],
    "open_interest": [1000000, 5000000, 10000000, 50000000],
    "price_ratio": [0.95, 1.05],
}

# TP/SL関連設定
GA_DEFAULT_TPSL_METHOD_CONSTRAINTS = [
    "fixed_percentage",
    "risk_reward_ratio",
    "volatility_based",
    "statistical",
    "adaptive",
]

GA_TPSL_SL_RANGE = [0.01, 0.08]  # SL範囲（1%-8%）
GA_TPSL_TP_RANGE = [0.02, 0.20]  # TP範囲（2%-20%）
GA_TPSL_RR_RANGE = [1.2, 4.0]   # リスクリワード比範囲
GA_TPSL_ATR_MULTIPLIER_RANGE = [1.0, 4.0]  # ATR倍率範囲

# ポジションサイジング関連設定
GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS = [
    "half_optimal_f",
    "volatility_based",
    "fixed_ratio",
    "fixed_quantity",
]

GA_POSITION_SIZING_LOOKBACK_RANGE = [50, 200]  # ハーフオプティマルF用ルックバック期間
GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE = [0.25, 0.75]  # オプティマルF倍率範囲
GA_POSITION_SIZING_ATR_PERIOD_RANGE = [10, 30]  # ATR計算期間範囲
GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE = [1.0, 4.0]  # ポジションサイジング用ATR倍率範囲
GA_POSITION_SIZING_RISK_PER_TRADE_RANGE = [0.01, 0.05]  # 1取引あたりのリスク範囲（1%-5%）
GA_POSITION_SIZING_FIXED_RATIO_RANGE = [0.05, 0.3]  # 固定比率範囲（5%-30%）
GA_POSITION_SIZING_FIXED_QUANTITY_RANGE = [0.1, 5.0]  # 固定枚数範囲
GA_POSITION_SIZING_MIN_SIZE_RANGE = [0.01, 0.1]  # 最小ポジションサイズ範囲
GA_POSITION_SIZING_MAX_SIZE_RANGE = [0.001, 1.0]  # 最大ポジションサイズ範囲（システム全体のmax_position_sizeに一致）
GA_POSITION_SIZING_PRIORITY_RANGE = [0.5, 1.5]  # 優先度範囲

# フィットネス共有設定
GA_DEFAULT_FITNESS_SHARING = {
    "enable_fitness_sharing": True,
    "sharing_radius": 0.1,
    "sharing_alpha": 1.0,
}
