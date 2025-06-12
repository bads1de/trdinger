"""
テクニカル指標パッケージ

分割されたテクニカル指標クラスと統合サービスを提供します。
"""

from typing import Dict, Any

from .indicator_orchestrator import TechnicalIndicatorService
from .abstract_indicator import BaseIndicator

# 各カテゴリの指標
from .trend_indicators import (
    SMAIndicator,
    EMAIndicator,
    MACDIndicator,
    KAMAIndicator,
    T3Indicator,
    TEMAIndicator,
    DEMAIndicator,
    WMAIndicator,
    HMAIndicator,
    VWMAIndicator,
    ZLEMAIndicator,
    get_trend_indicator,
    TREND_INDICATORS_INFO,
)

from .momentum_indicators import (
    RSIIndicator,
    StochasticIndicator,
    CCIIndicator,
    WilliamsRIndicator,
    MomentumIndicator,
    ROCIndicator,
    ADXIndicator,
    AroonIndicator,
    MFIIndicator,
    StochasticRSIIndicator,
    UltimateOscillatorIndicator,
    get_momentum_indicator,
    MOMENTUM_INDICATORS_INFO,
)

from .volatility_indicators import (
    BollingerBandsIndicator,
    ATRIndicator,
    NATRIndicator,
    TRANGEIndicator,
    KeltnerChannelsIndicator,
    STDDEVIndicator,
    DonchianChannelsIndicator,
    get_volatility_indicator,
    VOLATILITY_INDICATORS_INFO,
)

from .volume_indicators import (
    OBVIndicator,
    ADIndicator,
    ADOSCIndicator,
    VWAPIndicator,
    PVTIndicator,
    EMVIndicator,
    get_volume_indicator,
    VOLUME_INDICATORS_INFO,
)

from .other_indicators import PSARIndicator, get_other_indicator, OTHER_INDICATORS_INFO

# 公開API
__all__ = [
    # メインサービス
    "TechnicalIndicatorService",
    "BaseIndicator",
    # トレンド系指標
    "SMAIndicator",
    "EMAIndicator",
    "MACDIndicator",
    "KAMAIndicator",
    "T3Indicator",
    "TEMAIndicator",
    "DEMAIndicator",
    "WMAIndicator",
    "HMAIndicator",
    "VWMAIndicator",
    "ZLEMAIndicator",
    "get_trend_indicator",
    "TREND_INDICATORS_INFO",
    # モメンタム系指標
    "RSIIndicator",
    "StochasticIndicator",
    "CCIIndicator",
    "WilliamsRIndicator",
    "MomentumIndicator",
    "ROCIndicator",
    "ADXIndicator",
    "AroonIndicator",
    "MFIIndicator",
    "StochasticRSIIndicator",
    "UltimateOscillatorIndicator",
    "get_momentum_indicator",
    "MOMENTUM_INDICATORS_INFO",
    # ボラティリティ系指標
    "BollingerBandsIndicator",
    "ATRIndicator",
    "NATRIndicator",
    "TRANGEIndicator",
    "KeltnerChannelsIndicator",
    "STDDEVIndicator",
    "DonchianChannelsIndicator",
    "get_volatility_indicator",
    "VOLATILITY_INDICATORS_INFO",
    # 出来高系指標
    "OBVIndicator",
    "ADIndicator",
    "ADOSCIndicator",
    "VWAPIndicator",
    "PVTIndicator",
    "EMVIndicator",
    "get_volume_indicator",
    "VOLUME_INDICATORS_INFO",
    # その他の指標
    "PSARIndicator",
    "get_other_indicator",
    "OTHER_INDICATORS_INFO",
]


# 全指標情報の統合
ALL_INDICATORS_INFO = {}
ALL_INDICATORS_INFO.update(TREND_INDICATORS_INFO)
ALL_INDICATORS_INFO.update(MOMENTUM_INDICATORS_INFO)
ALL_INDICATORS_INFO.update(VOLATILITY_INDICATORS_INFO)
ALL_INDICATORS_INFO.update(VOLUME_INDICATORS_INFO)
ALL_INDICATORS_INFO.update(OTHER_INDICATORS_INFO)


def get_indicator_by_type(indicator_type: str) -> BaseIndicator:
    """
    指標タイプに応じた指標インスタンスを取得

    Args:
        indicator_type: 指標タイプ

    Returns:
        指標インスタンス

    Raises:
        ValueError: サポートされていない指標タイプの場合
    """
    # カテゴリ別に適切なファクトリー関数を呼び出し
    trend_indicators = [
        "SMA",
        "EMA",
        "MACD",
        "KAMA",
        "T3",
        "TEMA",
        "DEMA",
        "WMA",
        "HMA",
        "VWMA",
        "ZLEMA",
    ]
    momentum_indicators = [
        "RSI",
        "STOCH",
        "CCI",
        "WILLR",
        "MOM",
        "ROC",
        "ADX",
        "AROON",
        "MFI",
        "STOCHRSI",
        "ULTOSC",
    ]
    volatility_indicators = [
        "BB",
        "ATR",
        "NATR",
        "TRANGE",
        "KELTNER",
        "STDDEV",
        "DONCHIAN",
    ]
    volume_indicators = ["OBV", "AD", "ADOSC", "VWAP", "PVT", "EMV"]
    other_indicators = ["PSAR"]

    if indicator_type in trend_indicators:
        return get_trend_indicator(indicator_type)
    elif indicator_type in momentum_indicators:
        return get_momentum_indicator(indicator_type)
    elif indicator_type in volatility_indicators:
        return get_volatility_indicator(indicator_type)
    elif indicator_type in volume_indicators:
        return get_volume_indicator(indicator_type)
    elif indicator_type in other_indicators:
        return get_other_indicator(indicator_type)
    else:
        raise ValueError(
            f"サポートされていない指標タイプです: {indicator_type}. "
            f"サポート対象: {list(ALL_INDICATORS_INFO.keys())}"
        )


def get_all_supported_indicators() -> Dict[str, Any]:
    """
    全てのサポートされている指標の情報を取得

    Returns:
        全指標の情報
    """
    return ALL_INDICATORS_INFO.copy()


def get_indicators_by_category(category: str) -> Dict[str, Any]:
    """
    カテゴリ別の指標情報を取得

    Args:
        category: カテゴリ名（'trend', 'momentum', 'volatility', 'volume', 'other'）

    Returns:
        カテゴリ別の指標情報

    Raises:
        ValueError: 無効なカテゴリの場合
    """
    category_mapping = {
        "trend": TREND_INDICATORS_INFO,
        "momentum": MOMENTUM_INDICATORS_INFO,
        "volatility": VOLATILITY_INDICATORS_INFO,
        "volume": VOLUME_INDICATORS_INFO,
        "other": OTHER_INDICATORS_INFO,
    }

    if category not in category_mapping:
        raise ValueError(
            f"無効なカテゴリです: {category}. "
            f"有効なカテゴリ: {list(category_mapping.keys())}"
        )

    return category_mapping[category].copy()
