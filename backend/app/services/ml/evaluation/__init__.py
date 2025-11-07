"""
ML Evaluation パッケージ

包括的な評価指標計算とメトリクス管理を提供します。
"""

from .metrics import (
    MetricsCalculator,
    MetricsConfig,
    MetricData,
    PerformanceMetrics,
    ModelEvaluationMetrics,
    record_metric,
    record_performance,
    record_error,
    record_model_evaluation_metrics,
    evaluate_and_record_model,
    MLMetricsCollector,
    metrics_collector,
)

__all__ = [
    # Core classes
    "MetricsCalculator",
    "MetricsConfig",
    "MetricData",
    "PerformanceMetrics",
    "ModelEvaluationMetrics",
    # Global instances
    "metrics_collector",
    "MLMetricsCollector",
    # Utility functions
    "record_metric",
    "record_performance",
    "record_error",
    "record_model_evaluation_metrics",
    "evaluate_and_record_model",
]
