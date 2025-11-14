"""
Auto Strategy Utils パッケージ

戦略生成・評価・統合に関連する共通ユーティリティを提供します。
"""

# Operand Grouping
from ..core.operand_grouping import (
    OperandGroup,
    OperandGroupingSystem,
    operand_grouping_system,
)

# Compatibility imports from data_converters
from .data_converters import DataConverter, safe_execute

# Core Utilities
from .gene_utils import (
    BaseGene,
    GeneticUtils,
    GeneUtils,
    create_child_metadata,
    create_default_strategy_gene,
    prepare_crossover_metadata,
)
from .indicator_characteristics import INDICATOR_CHARACTERISTICS

# Strategy Integration
from .strategy_integration_service import StrategyIntegrationService
from .yaml_utils import (
    MockIndicatorGene,
    YamlIndicatorUtils,
    YamlLoadUtils,
    YamlTestUtils,
)

# Error Handling - Removed as dead code


# Utility functions from data_converters
ensure_float = DataConverter.ensure_float
ensure_int = DataConverter.ensure_int
ensure_dict = DataConverter.ensure_dict
normalize_symbol = DataConverter.normalize_symbol


__all__ = [
    # Core Utilities
    "BaseGene",
    "GeneticUtils",
    "GeneUtils",
    "DataConverter",
    "ValidationUtils",
    "YamlIndicatorUtils",
    "YamlLoadUtils",
    "YamlTestUtils",
    "MockIndicatorGene",
    # Utility functions
    "create_default_strategy_gene",
    "normalize_parameter",
    "create_child_metadata",
    "prepare_crossover_metadata",
    "safe_execute",
    "ensure_float",
    "ensure_int",
    "ensure_dict",
    "normalize_symbol",
    # Operand Grouping
    "OperandGroup",
    "OperandGroupingSystem",
    "operand_grouping_system",
    # Strategy Integration
    "StrategyIntegrationService",
    # Indicator Characteristics
    "INDICATOR_CHARACTERISTICS",
]
