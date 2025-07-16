"""
ML共通モジュール

ML関連サービスで共通して使用される機能を提供します。
"""

from .error_handler import (
    MLCommonErrorHandler,
    safe_ml_operation,
    validate_dataframe,
    validate_ml_predictions,
    validate_ml_indicators,
    handle_data_error,
    handle_prediction_error,
    handle_model_error,
    handle_validation_error,
    handle_timeout_error,
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
    "MLCommonErrorHandler",
    "safe_ml_operation",
    "validate_dataframe",
    "validate_ml_predictions",
    "validate_ml_indicators",
    "handle_data_error",
    "handle_prediction_error",
    "handle_model_error",
    "handle_validation_error",
    "handle_timeout_error",
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
