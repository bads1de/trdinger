"""
Auto Strategy Utils パッケージ

戦略生成・評価・統合に関連する共通ユーティリティを提供します。
"""


def __getattr__(name: str):
    """遅延インポートで循環インポートを回避"""
    if name in ("OperandGroup", "OperandGroupingSystem", "operand_grouping_system"):
        from ..core.strategy.operand_grouping import (
            OperandGroup,
            OperandGroupingSystem,
            operand_grouping_system,
        )

        return {
            "OperandGroup": OperandGroup,
            "OperandGroupingSystem": OperandGroupingSystem,
            "operand_grouping_system": operand_grouping_system,
        }[name]
    if name in ("NormalizationUtils", "create_default_strategy_gene"):
        from .normalization import (
            NormalizationUtils,
            create_default_strategy_gene,
        )

        return {
            "NormalizationUtils": NormalizationUtils,
            "create_default_strategy_gene": create_default_strategy_gene,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .normalization import (
    NormalizationUtils,
    create_default_strategy_gene,
)


__all__ = [
    # Core Utilities
    "NormalizationUtils",
    "OperandGroup",
    "OperandGroupingSystem",
    "operand_grouping_system",
    # Utility functions
    "create_default_strategy_gene",
]
