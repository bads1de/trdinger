"""
互換レイヤー: strategy_models

このモジュールは分割済みモデル群からの再エクスポートのみを提供します。
新規コードでは `from app.services.auto_strategy.models import ...` の利用を推奨します。
"""

from __future__ import annotations

# Core models
from .condition import Condition, ConditionGroup
from .indicator_gene import IndicatorGene
from .position_sizing_gene import PositionSizingGene
from .tpsl_gene import TPSLGene
from .strategy_gene import StrategyGene
from .tpsl_result import TPSLResult

# Enums
from .enums import PositionSizingMethod, TPSLMethod

# Validator
from .validator import GeneValidator

# Utilities
from .gene_random import (
    create_random_position_sizing_gene,
    create_random_tpsl_gene,
)
from .gene_crossover import (
    crossover_position_sizing_genes,
    crossover_tpsl_genes,
)
from .gene_mutation import (
    mutate_position_sizing_gene,
    mutate_tpsl_gene,
)

__all__ = [
    # Core models
    "StrategyGene",
    "IndicatorGene",
    "Condition",
    "ConditionGroup",
    "TPSLGene",
    "PositionSizingGene",
    "TPSLResult",
    # Enums
    "PositionSizingMethod",
    "TPSLMethod",
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

