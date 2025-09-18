"""
Auto Strategy Utils パッケージ

戦略生成・評価・統合に関連する共通ユーティリティを提供します。
"""

# Core Utilities
from .gene_utils import (
    BaseGene,
    GeneticUtils,
    GeneUtils,
    create_default_strategy_gene,
    normalize_parameter,
    create_child_metadata,
    prepare_crossover_metadata,
)
from .data_converters import DataConverter
from .logging_utils import LoggingUtils

from .yaml_utils import (
    YamlIndicatorUtils,
    YamlLoadUtils,
    YamlTestUtils,
    MockIndicatorGene,
)
from .indicator_characteristics import INDICATOR_CHARACTERISTICS

# Compatibility imports from data_converters
from .data_converters import safe_execute


# Error Handling - Removed as dead code

# Operand Grouping
from ..core.operand_grouping import (
    OperandGroup,
    OperandGroupingSystem,
    operand_grouping_system,
)

# Strategy Integration
from .strategy_integration_service import StrategyIntegrationService

# Utility functions from data_converters
ensure_float = DataConverter.ensure_float
ensure_int = DataConverter.ensure_int
ensure_list = DataConverter.ensure_list
ensure_dict = DataConverter.ensure_dict
normalize_symbol = DataConverter.normalize_symbol


__all__ = [
    # Core Utilities
    "BaseGene",
    "GeneticUtils",
    "GeneUtils",
    "DataConverter",
    "ValidationUtils",
    "LoggingUtils",
    "PerformanceUtils",
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
    "ensure_list",
    "ensure_dict",
    "normalize_symbol",
    "validate_range",
    "validate_required_fields",
    "time_function",
    # Decorators
    "safe_auto_operation",
    # Operand Grouping
    "OperandGroup",
    "OperandGroupingSystem",
    "operand_grouping_system",
    # Strategy Integration
    "StrategyIntegrationService",
    # Indicator Characteristics
    "INDICATOR_CHARACTERISTICS",
]
