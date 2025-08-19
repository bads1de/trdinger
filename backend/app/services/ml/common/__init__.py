"""
ML共通モジュール

ML関連サービスで共通して使用される機能を提供します。
"""

# ML例外クラスを直接インポート
from ..exceptions import (
    MLDataError,
    MLModelError,
    MLTimeoutError,
    MLValidationError,
)

# 統一エラーハンドリング機能をインポート
from ....utils.error_handler import (
    DataError,
    ErrorHandler,
    ModelError,
    TimeoutError,
    ValidationError,
    ml_operation_context,
    safe_ml_operation,
    timeout_decorator,
    safe_operation,
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
    "ErrorHandler",
    "safe_operation",
    "timeout_decorator",
    "operation_context",
    "TimeoutError",
    "ValidationError",
    "DataError",
    "ModelError",
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
    "record_model_evaluation_metrics",
    "ModelEvaluationMetrics",
    "get_metrics_summary",
]
