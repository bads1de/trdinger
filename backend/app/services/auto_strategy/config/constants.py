"""
Auto Strategy 共通定数

"""

from typing import Dict, List

# === 演算子定数 ===
OPERATORS = [
    ">",
    "<",
    ">=",
    "<=",
    "==",
    "!=",  # 基本比較演算子
    "above",
    "below",  # フロントエンド用演算子
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
    "BTC/USDT:USDT",  # Bitcoin USDT無期限先物
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
    # ボラティリティ系指標
    "ATR",
    "NATR",
    "TRANGE",
    "BB",
    "DONCHIAN",
    "KELTNER",
    "SUPERTREND",
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
    # 価格変換系指標
    "AVGPRICE",
    "MEDPRICE",
    "TYPPRICE",
    "WCLPRICE",
    "HA_CLOSE",
    "HA_OHLC",
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
    # 複合指標
    "ICHIMOKU",
    # 従来の指標（互換性維持）
    "BBANDS",  # BBの別名
    "CMO",  # 未実装だが互換性維持
    "DEMA",  # 未実装だが互換性維持
    "TEMA",  # 未実装だが互換性維持
    "MAMA",  # 未実装だが互換性維持
    "T3",  # 未実装だが互換性維持
    "UO",  # 未実装だが互換性維持
    "MOM",  # 未実装だが互換性維持
    "MAXINDEX",  # 未実装だが互換性維持
    "MININDEX",  # 未実装だが互換性維持
    "MINMAX",  # 未実装だが互換性維持
    "MINMAXINDEX",  # 未実装だが互換性維持
    "HT_DCPERIOD",  # 未実装だが互換性維持
    "HT_DCPHASE",  # 未実装だが互換性維持
    "HT_PHASOR",  # 未実装だが互換性維持
    "HT_SINE",  # 未実装だが互換性維持
    "HT_TRENDMODE",  # 未実装だが互換性維持
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
    auto_strategy_utils.py と gene_utils.py の重複機能を統合しています。
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
}
