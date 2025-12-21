"""
ML Evaluation パッケージ

包括的な評価指標計算とメトリクス管理を提供します。
"""

from .metrics import (
    MetricsCalculator,
    MetricsConfig,
    metrics_collector,
)

__all__ = [
    # Core classes
    "MetricsCalculator",
    "MetricsConfig",
    # Global instances
    "metrics_collector",
]