"""
インジケーター設定モジュール

動的検出されたインジケーター設定を管理します。
"""

from .indicator_config import (
    IndicatorConfig,
    IndicatorConfigRegistry,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
    indicator_registry,
    initialize_all_indicators,
)

__all__ = [
    "IndicatorConfig",
    "ParameterConfig",
    "IndicatorResultType",
    "IndicatorScaleType",
    "IndicatorConfigRegistry",
    "indicator_registry",
    "initialize_all_indicators",
]
