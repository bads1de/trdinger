"""
自動戦略生成パッケージ

遺伝的アルゴリズム（GA）を使用した取引戦略の自動生成機能を提供します。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

from ._lazy_import import setup_lazy_import  # noqa: E402
setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS, __all__)
