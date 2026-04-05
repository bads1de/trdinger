"""
Auto Strategy 向けインジケーターユニバース定義

GA の探索空間と、システム全体で計算可能なインジケーター集合を分離する。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, List


class IndicatorUniverseMode(str, Enum):
    """GA が使うインジケーターユニバースのモード。"""

    CURATED = "curated"
    EXPERIMENTAL_ALL = "experimental_all"


CURATED_INDICATOR_CATALOG: tuple[str, ...] = (
    # Trend / price
    "SMA",
    "EMA",
    "KAMA",
    "HMA",
    "ZLEMA",
    "T3",
    "ADX",
    "AROON",
    "SUPERTREND",
    "VORTEX",
    "CHOP",
    # Momentum
    "RSI",
    "MACD",
    "STOCH",
    "CCI",
    "TRIX",
    # Volatility / band
    "BBANDS",
    "ATR",
    # OI / FR derived
    "CRYPTO_LEVERAGE_INDEX",
    "LIQUIDATION_CASCADE_SCORE",
    "SQUEEZE_PROBABILITY",
    "TREND_QUALITY",
    "OI_WEIGHTED_FUNDING_RATE",
    "REGIME_QUADRANT",
    "WHALE_DIVERGENCE",
    "OI_PRICE_CONFIRMATION",
)

STANDARD_BACKTEST_REQUIRED_DATA = frozenset(
    {"open", "high", "low", "close", "volume", "open_interest", "funding_rate"}
)

_CONDITION_INCOMPATIBLE_CURATED_INDICATORS = frozenset(
    {
        # 4象限のカテゴリ値は現行の > / < 条件生成と相性が悪い
        "REGIME_QUADRANT",
    }
)


def normalize_indicator_universe_mode(value: Any) -> str:
    """ユニバースモードを正規化する。"""
    if isinstance(value, IndicatorUniverseMode):
        return value.value
    if value is None or value == "":
        return IndicatorUniverseMode.CURATED.value

    normalized = str(value).strip().lower()
    valid_values = {mode.value for mode in IndicatorUniverseMode}
    if normalized not in valid_values:
        raise ValueError(
            f"Unsupported indicator_universe_mode: {value!r}. "
            f"Expected one of: {', '.join(sorted(valid_values))}"
        )
    return normalized


def _extract_mode(config_or_mode: Any = None) -> str:
    if config_or_mode is None or isinstance(
        config_or_mode, (str, IndicatorUniverseMode)
    ):
        raw_value = config_or_mode
    else:
        candidate = getattr(config_or_mode, "indicator_universe_mode", None)
        if isinstance(candidate, (str, IndicatorUniverseMode)) or candidate in (
            None,
            "",
        ):
            raw_value = candidate
        else:
            raw_value = None
    return normalize_indicator_universe_mode(raw_value)


def _get_supported_indicator_names() -> set[str]:
    from app.services.indicators import TechnicalIndicatorService

    indicator_service = TechnicalIndicatorService()
    return set(indicator_service.get_supported_indicators().keys())


def _uses_standard_backtest_data(indicator_name: str) -> bool:
    from app.services.indicators.config import indicator_registry

    config = indicator_registry.get_indicator_config(indicator_name)
    if config is None:
        return False

    required_data = {source.lower() for source in (config.required_data or [])}
    return required_data.issubset(STANDARD_BACKTEST_REQUIRED_DATA)


def _supports_condition_generation(indicator_name: str) -> bool:
    return indicator_name not in _CONDITION_INCOMPATIBLE_CURATED_INDICATORS


def get_indicator_universe_names(config_or_mode: Any = None) -> List[str]:
    """
    指定モードで利用可能なインジケータ名一覧を返す。

    curated は固定 catalog を起点に、実装有無・データ可用性・条件生成互換を検証する。
    experimental_all は実装済みの全インジケーターを返す。
    """
    mode = _extract_mode(config_or_mode)
    supported_names = _get_supported_indicator_names()

    if mode == IndicatorUniverseMode.EXPERIMENTAL_ALL.value:
        return sorted(supported_names)

    curated_names: list[str] = []
    for indicator_name in CURATED_INDICATOR_CATALOG:
        if indicator_name not in supported_names:
            continue
        if not _uses_standard_backtest_data(indicator_name):
            continue
        if not _supports_condition_generation(indicator_name):
            continue
        curated_names.append(indicator_name)

    return curated_names


def is_indicator_in_universe(indicator_name: str, config_or_mode: Any = None) -> bool:
    """指標名が指定ユニバースに含まれるかを返す。"""
    if not indicator_name:
        return False
    return indicator_name.upper() in set(get_indicator_universe_names(config_or_mode))


def iter_indicator_universe_names(config_or_mode: Any = None) -> Iterable[str]:
    """ユニバース名一覧の iterable を返す。"""
    return tuple(get_indicator_universe_names(config_or_mode))
