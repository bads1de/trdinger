"""
自動戦略生成パッケージ

遺伝的アルゴリズム（GA）を使用した取引戦略の自動生成機能を提供します。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import GAConfig
    from .genes import StrategyGene
    from .services.auto_strategy_service import AutoStrategyService
    from .positions.position_sizing_service import PositionSizingService
    from .tpsl import TPSLService

_ATTRIBUTE_EXPORTS = {
    "AutoStrategyService": ".services.auto_strategy_service",
    "StrategyGene": ".genes",
    "GAConfig": ".config",
    "TPSLService": ".tpsl",
    "PositionSizingService": ".positions.position_sizing_service",
}

__all__ = [
    "AutoStrategyService",
    "StrategyGene",
    "GAConfig",
    "TPSLService",
    "PositionSizingService",
]


def __getattr__(name: str) -> Any:
    module_path = _ATTRIBUTE_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_ATTRIBUTE_EXPORTS})
