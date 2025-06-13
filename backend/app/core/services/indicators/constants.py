"""
テクニカル指標の共通定数

全指標リストを一元管理し、重複を防ぎます。
"""

from typing import List, Dict, Any

# 全テクニカル指標リスト（57個）
ALL_INDICATORS: List[str] = [
    # トレンド系指標（15個）
    "SMA",  # Simple Moving Average
    "EMA",  # Exponential Moving Average
    "WMA",  # Weighted Moving Average
    "HMA",  # Hull Moving Average
    "KAMA",  # Kaufman Adaptive Moving Average
    "TEMA",  # Triple Exponential Moving Average
    "DEMA",  # Double Exponential Moving Average
    "T3",  # T3 Moving Average
    "MAMA",  # MESA Adaptive Moving Average
    "ZLEMA",  # Zero Lag Exponential Moving Average
    "MACD",  # Moving Average Convergence Divergence
    "MIDPOINT",  # MidPoint over period
    "MIDPRICE",  # Midpoint Price over period
    "TRIMA",  # Triangular Moving Average
    "VWMA",  # Volume Weighted Moving Average
    # モメンタム系指標（24個）
    "RSI",  # Relative Strength Index
    "STOCH",  # Stochastic
    "STOCHRSI",  # Stochastic RSI
    "STOCHF",  # Stochastic Fast
    "CCI",  # Commodity Channel Index
    "WILLR",  # Williams %R
    "MOMENTUM",  # Momentum (別名)
    "MOM",  # Momentum (正式名)
    "ROC",  # Rate of Change
    "ROCP",  # Rate of change Percentage
    "ROCR",  # Rate of change ratio
    "ADX",  # Average Directional Movement Index
    "AROON",  # Aroon
    "AROONOSC",  # Aroon Oscillator
    "MFI",  # Money Flow Index
    "CMO",  # Chande Momentum Oscillator
    "TRIX",  # Triple Exponential Moving Average
    "ULTOSC",  # Ultimate Oscillator
    "BOP",  # Balance Of Power
    "APO",  # Absolute Price Oscillator
    "PPO",  # Percentage Price Oscillator
    "DX",  # Directional Movement Index
    "ADXR",  # Average Directional Movement Index Rating
    "PLUS_DI",  # Plus Directional Indicator
    "MINUS_DI",  # Minus Directional Indicator
    # ボラティリティ系指標（7個）
    "BB",  # Bollinger Bands
    "ATR",  # Average True Range
    "NATR",  # Normalized Average True Range
    "TRANGE",  # True Range
    "KELTNER",  # Keltner Channels
    "STDDEV",  # Standard Deviation
    "DONCHIAN",  # Donchian Channels
    # 出来高系指標（6個）
    "OBV",  # On Balance Volume
    "AD",  # Accumulation/Distribution Line
    "ADOSC",  # Accumulation/Distribution Oscillator
    "VWAP",  # Volume Weighted Average Price
    "PVT",  # Price Volume Trend
    "EMV",  # Ease of Movement
    # 価格変換系指標（4個）
    "AVGPRICE",  # Average Price
    "MEDPRICE",  # Median Price
    "TYPPRICE",  # Typical Price
    "WCLPRICE",  # Weighted Close Price
    # その他の指標（1個）
    "PSAR",  # Parabolic SAR
]

# カテゴリ別指標リスト
TREND_INDICATORS: List[str] = [
    "SMA",
    "EMA",
    "WMA",
    "HMA",
    "KAMA",
    "TEMA",
    "DEMA",
    "T3",
    "MAMA",
    "ZLEMA",
    "MACD",
    "MIDPOINT",
    "MIDPRICE",
    "TRIMA",
    "VWMA",
]

MOMENTUM_INDICATORS: List[str] = [
    "RSI",
    "STOCH",
    "STOCHRSI",
    "STOCHF",
    "CCI",
    "WILLR",
    "MOMENTUM",
    "MOM",
    "ROC",
    "ROCP",
    "ROCR",
    "ADX",
    "AROON",
    "AROONOSC",
    "MFI",
    "CMO",
    "TRIX",
    "ULTOSC",
    "BOP",
    "APO",
    "PPO",
    "DX",
    "ADXR",
    "PLUS_DI",
    "MINUS_DI",
]

VOLATILITY_INDICATORS: List[str] = [
    "BB",
    "ATR",
    "NATR",
    "TRANGE",
    "KELTNER",
    "STDDEV",
    "DONCHIAN",
]

VOLUME_INDICATORS: List[str] = ["OBV", "AD", "ADOSC", "VWAP", "PVT", "EMV"]

PRICE_TRANSFORM_INDICATORS: List[str] = ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]

OTHER_INDICATORS: List[str] = ["PSAR"]

# 指標情報辞書
INDICATOR_INFO: Dict[str, Dict[str, Any]] = {
    # トレンド系
    "SMA": {"name": "Simple Moving Average", "category": "trend", "min_period": 2},
    "EMA": {"name": "Exponential Moving Average", "category": "trend", "min_period": 2},
    "WMA": {"name": "Weighted Moving Average", "category": "trend", "min_period": 2},
    "HMA": {"name": "Hull Moving Average", "category": "trend", "min_period": 2},
    "KAMA": {
        "name": "Kaufman Adaptive Moving Average",
        "category": "trend",
        "min_period": 2,
    },
    "TEMA": {
        "name": "Triple Exponential Moving Average",
        "category": "trend",
        "min_period": 2,
    },
    "DEMA": {
        "name": "Double Exponential Moving Average",
        "category": "trend",
        "min_period": 2,
    },
    "T3": {"name": "T3 Moving Average", "category": "trend", "min_period": 2},
    "MAMA": {
        "name": "MESA Adaptive Moving Average",
        "category": "trend",
        "min_period": 2,
    },
    "ZLEMA": {
        "name": "Zero Lag Exponential Moving Average",
        "category": "trend",
        "min_period": 2,
    },
    "MACD": {
        "name": "Moving Average Convergence Divergence",
        "category": "trend",
        "min_period": 2,
    },
    "MIDPOINT": {"name": "MidPoint over period", "category": "trend", "min_period": 2},
    "MIDPRICE": {
        "name": "Midpoint Price over period",
        "category": "trend",
        "min_period": 2,
    },
    "TRIMA": {
        "name": "Triangular Moving Average",
        "category": "trend",
        "min_period": 2,
    },
    "VWMA": {
        "name": "Volume Weighted Moving Average",
        "category": "trend",
        "min_period": 2,
    },
    # モメンタム系
    "RSI": {"name": "Relative Strength Index", "category": "momentum", "min_period": 2},
    "STOCH": {"name": "Stochastic", "category": "momentum", "min_period": 2},
    "STOCHRSI": {"name": "Stochastic RSI", "category": "momentum", "min_period": 2},
    "STOCHF": {"name": "Stochastic Fast", "category": "momentum", "min_period": 2},
    "CCI": {"name": "Commodity Channel Index", "category": "momentum", "min_period": 2},
    "WILLR": {"name": "Williams %R", "category": "momentum", "min_period": 2},
    "MOMENTUM": {"name": "Momentum", "category": "momentum", "min_period": 1},
    "MOM": {"name": "Momentum", "category": "momentum", "min_period": 1},
    "ROC": {"name": "Rate of Change", "category": "momentum", "min_period": 1},
    "ROCP": {
        "name": "Rate of change Percentage",
        "category": "momentum",
        "min_period": 1,
    },
    "ROCR": {"name": "Rate of change ratio", "category": "momentum", "min_period": 1},
    "ADX": {
        "name": "Average Directional Movement Index",
        "category": "momentum",
        "min_period": 2,
    },
    "AROON": {"name": "Aroon", "category": "momentum", "min_period": 2},
    "AROONOSC": {"name": "Aroon Oscillator", "category": "momentum", "min_period": 2},
    "MFI": {"name": "Money Flow Index", "category": "momentum", "min_period": 2},
    "CMO": {
        "name": "Chande Momentum Oscillator",
        "category": "momentum",
        "min_period": 2,
    },
    "TRIX": {
        "name": "Triple Exponential Moving Average",
        "category": "momentum",
        "min_period": 2,
    },
    "ULTOSC": {"name": "Ultimate Oscillator", "category": "momentum", "min_period": 2},
    "BOP": {"name": "Balance Of Power", "category": "momentum", "min_period": 1},
    "APO": {
        "name": "Absolute Price Oscillator",
        "category": "momentum",
        "min_period": 2,
    },
    "PPO": {
        "name": "Percentage Price Oscillator",
        "category": "momentum",
        "min_period": 2,
    },
    "DX": {
        "name": "Directional Movement Index",
        "category": "momentum",
        "min_period": 2,
    },
    "ADXR": {
        "name": "Average Directional Movement Index Rating",
        "category": "momentum",
        "min_period": 2,
    },
    "PLUS_DI": {
        "name": "Plus Directional Indicator",
        "category": "momentum",
        "min_period": 2,
    },
    "MINUS_DI": {
        "name": "Minus Directional Indicator",
        "category": "momentum",
        "min_period": 2,
    },
    # ボラティリティ系
    "BB": {"name": "Bollinger Bands", "category": "volatility", "min_period": 2},
    "ATR": {"name": "Average True Range", "category": "volatility", "min_period": 2},
    "NATR": {
        "name": "Normalized Average True Range",
        "category": "volatility",
        "min_period": 2,
    },
    "TRANGE": {"name": "True Range", "category": "volatility", "min_period": 1},
    "KELTNER": {"name": "Keltner Channels", "category": "volatility", "min_period": 2},
    "STDDEV": {"name": "Standard Deviation", "category": "volatility", "min_period": 2},
    "DONCHIAN": {
        "name": "Donchian Channels",
        "category": "volatility",
        "min_period": 2,
    },
    # 出来高系
    "OBV": {"name": "On Balance Volume", "category": "volume", "min_period": 1},
    "AD": {
        "name": "Accumulation/Distribution Line",
        "category": "volume",
        "min_period": 1,
    },
    "ADOSC": {
        "name": "Accumulation/Distribution Oscillator",
        "category": "volume",
        "min_period": 2,
    },
    "VWAP": {
        "name": "Volume Weighted Average Price",
        "category": "volume",
        "min_period": 1,
    },
    "PVT": {"name": "Price Volume Trend", "category": "volume", "min_period": 1},
    "EMV": {"name": "Ease of Movement", "category": "volume", "min_period": 2},
    # 価格変換系
    "AVGPRICE": {
        "name": "Average Price",
        "category": "price_transform",
        "min_period": 1,
    },
    "MEDPRICE": {
        "name": "Median Price",
        "category": "price_transform",
        "min_period": 1,
    },
    "TYPPRICE": {
        "name": "Typical Price",
        "category": "price_transform",
        "min_period": 1,
    },
    "WCLPRICE": {
        "name": "Weighted Close Price",
        "category": "price_transform",
        "min_period": 1,
    },
    # その他
    "PSAR": {"name": "Parabolic SAR", "category": "other", "min_period": 2},
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
# 実際の指標数に合わせて調整（58個が正しい）
assert TOTAL_INDICATORS == 58, f"Expected 58 indicators, got {TOTAL_INDICATORS}"
assert validate_indicator_lists(), "Indicator lists are inconsistent"
