"""
ML共通モジュール

ML関連サービスで共通して使用される機能を提供します。
"""

from ....utils.unified_error_handler import (  # 標準エイリアス
    MLDataError,
    MLModelError,
    MLTimeoutError,
    MLValidationError,
    UnifiedDataError,
    UnifiedErrorHandler,
    UnifiedModelError,
    UnifiedTimeoutError,
    UnifiedValidationError,
    ml_operation_context,
    safe_ml_operation,
    timeout_decorator,
    unified_operation_context,
    unified_safe_operation,
    unified_timeout_decorator,
)

from .logger import (
    MLStructuredLogger,
    log_error,
    log_info,
    log_ml_data_summary,
    log_ml_metrics,
    log_ml_operation,
    log_warning,
    ml_logger,
)

# メトリクス機能は evaluation/enhanced_metrics.py に統合されました
# 後方互換性のため、enhanced_metrics からインポート
from ..evaluation.enhanced_metrics import (
    MetricData,
    ModelEvaluationMetrics,
    PerformanceMetrics,
    enhanced_metrics_calculator as metrics_collector,
    record_error,
    record_metric,
    record_model_evaluation_metrics,
    record_performance,
    # 後方互換性エイリアス
    MLMetricsCollector,
)

__all__ = [
    # Unified Error handling
    "UnifiedErrorHandler",
    "unified_safe_operation",
    "unified_timeout_decorator",
    "unified_operation_context",
    "UnifiedTimeoutError",
    "UnifiedValidationError",
    "UnifiedDataError",
    "UnifiedModelError",
    # Standard aliases
    "safe_ml_operation",
    "timeout_decorator",
    "ml_operation_context",
    "MLTimeoutError",
    "MLValidationError",
    "MLDataError",
    "MLModelError",
    # Logging
    "MLStructuredLogger",
    "log_ml_operation",
    "log_ml_metrics",
    "log_ml_data_summary",
    "ml_logger",
    "log_info",
    "log_warning",
    "log_error",
    # Metrics
    "MLMetricsCollector",
    "MetricData",
    "PerformanceMetrics",
    "performance_monitor",
    "metrics_collector",
    "record_metric",
    "record_performance",
    "record_error",
    "get_metrics_summary",
]
