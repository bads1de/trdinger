"""
ML共通モジュール

ML関連サービスで共通して使用される機能を提供します。
"""

# 統一エラーハンドリング機能をインポート
from ....utils.error_handler import (
    DataError,
    ErrorHandler,
    ModelError,
    TimeoutError,
    ValidationError,
    ml_operation_context,
    safe_ml_operation,
    safe_operation,
    timeout_decorator,
)

# メトリクス機能は evaluation/metrics.py に統合されています
# MetricsCalculator ベースのメトリクス機能を提供
from ..evaluation.metrics import (
    MetricData,
    ModelEvaluationMetrics,
    PerformanceMetrics,
    metrics_collector,
)

# ML例外クラスを直接インポート
from ..exceptions import (
    MLDataError,
    MLModelError,
    MLValidationError,
)
from .logger import (
    MLStructuredLogger,
    ml_logger,
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
    "MLValidationError",
    "MLDataError",
    "MLModelError",
    # Logging
    "MLStructuredLogger",
    "ml_logger",
    # Metrics
    "MetricData",
    "PerformanceMetrics",
    "metrics_collector",
    "ModelEvaluationMetrics",
]
