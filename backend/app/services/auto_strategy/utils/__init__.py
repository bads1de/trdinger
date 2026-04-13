"""
Auto Strategy Utils パッケージ

戦略生成・評価・統合に関連する共通ユーティリティを提供します。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.strategy.operand_grouping import (
        OperandGroup,
        OperandGroupingSystem,
        operand_grouping_system,
    )
    from .normalization import NormalizationUtils, create_default_strategy_gene

_ATTRIBUTE_EXPORTS = {
    "OperandGroup": "..core.strategy.operand_grouping",
    "OperandGroupingSystem": "..core.strategy.operand_grouping",
    "operand_grouping_system": "..core.strategy.operand_grouping",
    "NormalizationUtils": ".normalization",
    "create_default_strategy_gene": ".normalization",
}

__all__ = [
    # Core Utilities
    "NormalizationUtils",
    "OperandGroup",
    "OperandGroupingSystem",
    "operand_grouping_system",
    # Utility functions
    "create_default_strategy_gene",
]

from .._lazy_import import setup_lazy_import  # noqa: E402
setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS, __all__)
