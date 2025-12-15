"""
自動戦略生成モデル

戦略遺伝子、GA設定、フィットネス評価などのモデルを定義します。
"""

from .conditions import (
    Condition,
    ConditionGroup,
    EntryDirection,
    StateTracker,
    StatefulCondition,
)
from .entry_gene import EntryGene
from ..config.enums import EntryType, PositionSizingMethod, TPSLMethod
from .indicator_gene import IndicatorGene
from .position_sizing_gene import (
    PositionSizingGene,
    create_random_position_sizing_gene,
    crossover_position_sizing_genes,
    mutate_position_sizing_gene,
)
from .strategy_gene import StrategyGene
from .tool_gene import ToolGene
from .tpsl_gene import (
    TPSLGene,
    TPSLResult,
    create_random_tpsl_gene,
    crossover_tpsl_genes,
    mutate_tpsl_gene,
)
from .validator import GeneValidator

__all__ = [
    # Core models
    "StrategyGene",
    "IndicatorGene",
    "Condition",
    "ConditionGroup",
    "StatefulCondition",
    "StateTracker",
    "EntryDirection",
    "TPSLGene",
    "PositionSizingGene",
    "TPSLResult",
    "EntryGene",
    "ToolGene",
    # Enums
    "PositionSizingMethod",
    "TPSLMethod",
    "EntryType",
    # Validator
    "GeneValidator",
    # Utilities
    "create_random_position_sizing_gene",
    "create_random_tpsl_gene",
    "crossover_position_sizing_genes",
    "crossover_tpsl_genes",
    "mutate_position_sizing_gene",
    "mutate_tpsl_gene",
]


