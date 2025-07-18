"""
ML共通モジュール

ML関連サービスで共通して使用される機能を提供します。
"""

from app.core.utils.ml_error_handler import (
    MLErrorHandler,
    safe_ml_operation,
    timeout_decorator,
    ml_operation_context,
    MLTimeoutError,
    MLValidationError,
    MLDataError,
    MLModelError,
)

from .logger import (
    MLStructuredLogger,
    log_ml_operation,
    log_ml_metrics,
    log_ml_data_summary,
    ml_logger,
    log_info,
    log_warning,
    log_error,
    log_debug,
)

from .metrics import (
    MLMetricsCollector,
    MetricData,
    PerformanceMetrics,
    performance_monitor,
    metrics_collector,
    record_metric,
    record_performance,
    record_error,
    get_metrics_summary,
)

__all__ = [
    # Error handling
    "MLErrorHandler",
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
    "log_debug",
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
