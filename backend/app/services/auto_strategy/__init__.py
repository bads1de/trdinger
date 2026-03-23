"""
自動戦略生成パッケージ

遺伝的アルゴリズム（GA）を使用した取引戦略の自動生成機能を提供します。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AutoStrategyService",
    "StrategyGene",
    "GAConfig",
    "TPSLService",
    "PositionSizingService",
]


def __getattr__(name: str) -> Any:
    if name == "AutoStrategyService":
        from .services.auto_strategy_service import AutoStrategyService

        return AutoStrategyService
    if name == "StrategyGene":
        from .genes import StrategyGene

        return StrategyGene
    if name == "GAConfig":
        from .config import GAConfig

        return GAConfig
    if name == "TPSLService":
        from .tpsl import TPSLService

        return TPSLService
    if name == "PositionSizingService":
        from .positions import PositionSizingService

        return PositionSizingService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

