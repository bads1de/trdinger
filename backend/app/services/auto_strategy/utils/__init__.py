"""
Auto Strategy Utils パッケージ

戦略生成・評価・統合に関連する共通ユーティリティを提供します。
"""

# Core Utilities
from .common_utils import (
    GeneUtils,
    BaseGene,
    DataConverter,
    ValidationUtils,
    PerformanceUtils,
    LoggingUtils,
    CacheUtils,
    YamlLoadUtils,
    YamlTestUtils,
    create_default_strategy_gene,
    normalize_parameter,
)

# Data Coverage
from .data_coverage_analyzer import DataCoverageAnalyzer

# Decorators
from .decorators import (
    auto_strategy_operation,
    safe_auto_operation,
    with_metrics_tracking,
)

# Error Handling
from .error_handling import (
    AutoStrategyErrorHandler,
    ErrorContext,
)

# Metrics and Statistics
from .metrics import (
    SuccessStats,
    QualityMetrics,
    aggregate_success,
    score_strategy_quality,
    passes_quality_threshold,
    filter_and_rank_strategies,
)

# Operand Grouping
from ..core.operand_grouping import (
    OperandGroup,
    OperandGroupingSystem,
    operand_grouping_system,
)

# Strategy Integration
from .strategy_integration_service import StrategyIntegrationService

# Utility functions from common_utils
ensure_float = DataConverter.ensure_float
ensure_int = DataConverter.ensure_int
ensure_list = DataConverter.ensure_list
ensure_dict = DataConverter.ensure_dict
normalize_symbol = DataConverter.normalize_symbol
validate_range = ValidationUtils.validate_range
validate_required_fields = ValidationUtils.validate_required_fields
time_function = PerformanceUtils.time_function

__all__ = [
    # Core Utilities
    "GeneUtils",
    "BaseGene",
    "DataConverter",
    "ValidationUtils",
    "PerformanceUtils",
    "LoggingUtils",
    "CacheUtils",
    "YamlLoadUtils",
    "YamlTestUtils",
    "create_default_strategy_gene",
    "normalize_parameter",

    # Data Coverage
    "DataCoverageAnalyzer",

    # Decorators
    "auto_strategy_operation",
    "safe_auto_operation",
    "with_metrics_tracking",

    # Error Handling
    "AutoStrategyErrorHandler",
    "ErrorContext",

    # Metrics and Statistics
    "SuccessStats",
    "QualityMetrics",
    "aggregate_success",
    "score_strategy_quality",
    "passes_quality_threshold",
    "filter_and_rank_strategies",

    # Operand Grouping
    "OperandGroup",
    "OperandGroupingSystem",
    "operand_grouping_system",

    # Strategy Integration
    "StrategyIntegrationService",

    # Utility functions
    "ensure_float",
    "ensure_int",
    "ensure_list",
    "ensure_dict",
    "normalize_symbol",
    "validate_range",
    "validate_required_fields",
    "time_function",
]