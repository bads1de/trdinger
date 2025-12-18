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
from .entry import EntryGene, create_random_entry_gene
from ..config.constants import EntryType, PositionSizingMethod, TPSLMethod
from .indicator import IndicatorGene, generate_random_indicators, create_random_indicator_gene
from .position_sizing import (
    PositionSizingGene,
    create_random_position_sizing_gene,
)
from .strategy import StrategyGene
from .tool import ToolGene
from .tpsl import (
    TPSLGene,
    TPSLResult,
    create_random_tpsl_gene,
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
    "create_random_entry_gene",
    "generate_random_indicators",
    "create_random_indicator_gene",
]
