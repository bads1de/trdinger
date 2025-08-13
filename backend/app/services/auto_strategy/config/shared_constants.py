"""
Auto Strategy 共通定数

Auto Strategy全体で使用される定数を統一管理します。
フロントエンドとバックエンドで共有される定数も含みます。
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
VALID_INDICATOR_TYPES = [
    # トレンド系指標
    "SMA",
    "EMA",
    "WMA",
    "DEMA",
    "TEMA",
    "TRIMA",
    "KAMA",
    "MAMA",
    "T3",
    # モメンタム系指標
    "RSI",
    "STOCH",
    "STOCHF",
    "STOCHRSI",
    "MACD",
    "MACDEXT",
    "MACDFIX",
    "PPO",
    "APO",
    "CMO",
    "ROC",
    "ROCP",
    "ROCR",
    "ROCR100",
    "MOM",
    "TSI",
    "UO",
    "WILLR",
    "CCI",
    "DX",
    "MINUS_DI",
    "PLUS_DI",
    "MINUS_DM",
    "PLUS_DM",
    "ADX",
    "ADXR",
    "AROON",
    "AROONOSC",
    "BOP",
    "MFI",
    "TRIX",
    "RVGI",
    # ボラティリティ系指標
    "ATR",
    "NATR",
    "TRANGE",
    "BBANDS",
    "BB",  # ボリンジャーバンドの別名
    "KELTNER",
    "DONCHIAN",
    # ボリューム系指標
    "AD",
    "ADOSC",
    "OBV",
    # サイクル系指標
    "HT_DCPERIOD",
    "HT_DCPHASE",
    "HT_PHASOR",
    "HT_SINE",
    "HT_TRENDMODE",
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
    # 数学演算子
    "ADD",
    "DIV",
    "MULT",
    "SUB",
    "MAX",
    "MIN",
    "MAXINDEX",
    "MININDEX",
    "SUM",
    "MINMAX",
    "MINMAXINDEX",
    # 価格変換系指標
    "AVGPRICE",
    "MEDPRICE",
    "TYPPRICE",
    "WCLPRICE",
    # オーバーレイ系指標
    "SAR",
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


def get_indicator_categories() -> Dict[str, List[str]]:
    """指標をカテゴリ別に分類"""
    return {
        "trend": ["SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MAMA", "T3"],
        "momentum": [
            "RSI",
            "STOCH",
            "STOCHF",
            "STOCHRSI",
            "MACD",
            "MACDEXT",
            "MACDFIX",
        ],
        "volatility": ["ATR", "NATR", "TRANGE", "BBANDS", "KELTNER", "DONCHIAN"],
        "volume": ["AD", "ADOSC", "OBV"],
        "cycle": ["HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDMODE"],
        "statistical": ["BETA", "CORREL", "LINEARREG", "STDDEV", "TSF", "VAR"],
        "pattern": ["CDL_DOJI", "CDL_HAMMER", "CDL_HANGING_MAN", "CDL_SHOOTING_STAR"],
        "ml": ML_INDICATOR_TYPES,
    }


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


def validate_indicator_type(indicator_type: str) -> bool:
    """指標タイプの妥当性を検証"""
    return indicator_type in get_all_indicators()
