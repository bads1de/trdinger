"""
テクニカル指標の共通定数

全指標リストを一元管理し、重複を防ぎます。
"""

from typing import List, Dict, Any

# オートストラテジー用テクニカル指標リスト（10個）
# 効果的な戦略生成のために厳選された指標
ALL_INDICATORS: List[str] = [
    # トレンド系指標（4個）
    "SMA",  # Simple Moving Average - 最も基本的なトレンド指標
    "EMA",  # Exponential Moving Average - より反応の早いトレンド指標
    "MACD",  # Moving Average Convergence Divergence - トレンド転換を捉える
    "BB",  # Bollinger Bands - トレンドとボラティリティの両方を分析
    # モメンタム系指標（4個）
    "RSI",  # Relative Strength Index - 最も一般的なオシレーター
    "STOCH",  # Stochastic - 買われすぎ/売られすぎを判定
    "CCI",  # Commodity Channel Index - サイクル分析
    "ADX",  # Average Directional Movement Index - トレンドの強さを測定
    # ボラティリティ系指標（1個）
    "ATR",  # Average True Range - リスク管理に重要
    # 出来高系指標（1個）
    "OBV",  # On Balance Volume - 価格と出来高の関係を分析
]

# カテゴリ別指標リスト（オートストラテジー用）
TREND_INDICATORS: List[str] = [
    "SMA",  # Simple Moving Average
    "EMA",  # Exponential Moving Average
    "MACD",  # Moving Average Convergence Divergence
    "BB",  # Bollinger Bands
]

MOMENTUM_INDICATORS: List[str] = [
    "RSI",  # Relative Strength Index
    "STOCH",  # Stochastic
    "CCI",  # Commodity Channel Index
    "ADX",  # Average Directional Movement Index
]

VOLATILITY_INDICATORS: List[str] = [
    "ATR",  # Average True Range
]

VOLUME_INDICATORS: List[str] = [
    "OBV",  # On Balance Volume
]

# 価格変換系とその他の指標は使用しない（空リスト）
PRICE_TRANSFORM_INDICATORS: List[str] = []
OTHER_INDICATORS: List[str] = []

# 指標情報辞書（オートストラテジー用10個の指標のみ）
INDICATOR_INFO: Dict[str, Dict[str, Any]] = {
    # トレンド系指標（4個）
    "SMA": {"name": "Simple Moving Average", "category": "trend", "min_period": 2},
    "EMA": {"name": "Exponential Moving Average", "category": "trend", "min_period": 2},
    "MACD": {
        "name": "Moving Average Convergence Divergence",
        "category": "trend",
        "min_period": 2,
    },
    "BB": {"name": "Bollinger Bands", "category": "volatility", "min_period": 2},
    # モメンタム系指標（4個）
    "RSI": {"name": "Relative Strength Index", "category": "momentum", "min_period": 2},
    "STOCH": {"name": "Stochastic", "category": "momentum", "min_period": 2},
    "CCI": {"name": "Commodity Channel Index", "category": "momentum", "min_period": 2},
    "ADX": {
        "name": "Average Directional Movement Index",
        "category": "momentum",
        "min_period": 2,
    },
    # ボラティリティ系指標（1個）
    "ATR": {"name": "Average True Range", "category": "volatility", "min_period": 2},
    # 出来高系指標（1個）
    "OBV": {"name": "On Balance Volume", "category": "volume", "min_period": 1},
}


# 検証関数
def validate_indicator_lists() -> bool:
    """指標リストの整合性を検証"""
    category_total = (
        len(TREND_INDICATORS)
        + len(MOMENTUM_INDICATORS)
        + len(VOLATILITY_INDICATORS)
        + len(VOLUME_INDICATORS)
        + len(PRICE_TRANSFORM_INDICATORS)
        + len(OTHER_INDICATORS)
    )

    return len(ALL_INDICATORS) == category_total == len(INDICATOR_INFO)


# 指標数の確認
TOTAL_INDICATORS = len(ALL_INDICATORS)
# オートストラテジー用に10個に絞り込み
assert TOTAL_INDICATORS == 10, f"Expected 10 indicators, got {TOTAL_INDICATORS}"
assert validate_indicator_lists(), "Indicator lists are inconsistent"
