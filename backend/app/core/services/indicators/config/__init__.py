"""
インジケーター設定モジュール

JSON形式でのインジケーター設定管理を提供します。
"""

from .indicator_config import (
    IndicatorConfig,
    ParameterConfig,
    IndicatorResultType,
    IndicatorConfigRegistry,
    indicator_registry,
)

from .indicator_definitions import initialize_all_indicators

# 自動初期化
initialize_all_indicators()

__all__ = [
    "IndicatorConfig",
    "ParameterConfig",
    "IndicatorResultType",
    "IndicatorConfigRegistry",
    "indicator_registry",
    "initialize_all_indicators",
]
