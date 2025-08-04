"""
ML統一メトリクス収集機能

ML関連サービス共通のメトリクス収集とレポート機能を提供します。
"""

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

from .logger import ml_logger


@dataclass
class MetricData:
    """メトリクスデータ"""

    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""

    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    success: bool
    timestamp: datetime
    error_message: Optional[str] = None


class MLMetricsCollector:
    """ML統一メトリクス収集クラス"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._performance_metrics: deque = deque(maxlen=max_history)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._operation_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        メトリクスを記録

        Args:
            name: メトリクス名
            value: 値
            tags: タグ
            context: コンテキスト情報
        """
        with self._lock:
            metric = MetricData(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                context=context or {},
            )
            self._metrics[name].append(metric)

    def record_performance(
        self,
        operation: str,
        duration_ms: float,
        memory_mb: float = 0.0,
        cpu_percent: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """
        パフォーマンスメトリクスを記録

        Args:
            operation: 操作名
            duration_ms: 処理時間（ミリ秒）
            memory_mb: メモリ使用量（MB）
            cpu_percent: CPU使用率（%）
            success: 成功フラグ
            error_message: エラーメッセージ
        """
        with self._lock:
            perf_metric = PerformanceMetrics(
                operation=operation,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                success=success,
                timestamp=datetime.now(),
                error_message=error_message,
            )
            self._performance_metrics.append(perf_metric)

            # 操作カウント
            self._operation_counts[operation] += 1

            # エラーカウント
            if not success:
                self._error_counts[operation] += 1

    def record_error(self, operation: str, error_type: str, error_message: str):
        """
        エラーを記録

        Args:
            operation: 操作名
            error_type: エラータイプ
            error_message: エラーメッセージ
        """
        with self._lock:
            self._error_counts[f"{operation}_{error_type}"] += 1

        # ログにも出力
        ml_logger.log_error(
            operation=operation,
            error=Exception(error_message),
            context={"error_type": error_type},
        )

    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        メトリクスサマリーを取得

        Args:
            time_window_minutes: 時間窓（分）

        Returns:
            メトリクスサマリー
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

            summary = {
                "time_window_minutes": time_window_minutes,
                "timestamp": datetime.now().isoformat(),
                "performance": self._get_performance_summary(cutoff_time),
                "errors": self._get_error_summary(cutoff_time),
                "operations": dict(self._operation_counts),
                "custom_metrics": self._get_custom_metrics_summary(cutoff_time),
            }

            return summary

    def _get_performance_summary(self, cutoff_time: datetime) -> Dict[str, Any]:
        """パフォーマンスサマリーを取得"""
        recent_metrics = [
            m for m in self._performance_metrics if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {}

        # 操作別集計
        by_operation = defaultdict(list)
        for metric in recent_metrics:
            by_operation[metric.operation].append(metric)

        summary = {}
        for operation, metrics in by_operation.items():
            durations = [m.duration_ms for m in metrics]
            memory_usage = [m.memory_mb for m in metrics if m.memory_mb > 0]
            success_count = sum(1 for m in metrics if m.success)

            summary[operation] = {
                "count": len(metrics),
                "success_rate": success_count / len(metrics) if metrics else 0,
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0,
                "min_duration_ms": min(durations) if durations else 0,
                "avg_memory_mb": (
                    sum(memory_usage) / len(memory_usage) if memory_usage else 0
                ),
                "max_memory_mb": max(memory_usage) if memory_usage else 0,
            }

        return summary

    def _get_error_summary(self, cutoff_time: datetime) -> Dict[str, Any]:
        """エラーサマリーを取得"""
        recent_errors = [
            m
            for m in self._performance_metrics
            if m.timestamp >= cutoff_time and not m.success
        ]

        error_by_operation = defaultdict(int)
        error_by_type = defaultdict(int)

        for error in recent_errors:
            error_by_operation[error.operation] += 1
            if error.error_message:
                error_type = type(Exception(error.error_message)).__name__
                error_by_type[error_type] += 1

        return {
            "total_errors": len(recent_errors),
            "by_operation": dict(error_by_operation),
            "by_type": dict(error_by_type),
            "error_rate": (
                len(recent_errors) / len(self._performance_metrics)
                if self._performance_metrics
                else 0
            ),
        }

    def _get_custom_metrics_summary(self, cutoff_time: datetime) -> Dict[str, Any]:
        """カスタムメトリクスサマリーを取得"""
        summary = {}

        for metric_name, metrics in self._metrics.items():
            recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary[metric_name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                    "latest": values[-1] if values else None,
                }

        return summary

    def export_metrics(self, format: str = "json") -> str:
        """
        メトリクスをエクスポート

        Args:
            format: エクスポート形式（json, csv）

        Returns:
            エクスポートされたデータ
        """
        summary = self.get_metrics_summary()

        if format.lower() == "json":
            return json.dumps(summary, indent=2, ensure_ascii=False)
        elif format.lower() == "csv":
            # CSV形式での出力（簡易版）
            lines = ["metric_name,value,timestamp"]
            for metric_name, metrics in self._metrics.items():
                for metric in metrics:
                    lines.append(
                        f"{metric_name},{metric.value},{metric.timestamp.isoformat()}"
                    )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear_metrics(self, older_than_hours: int = 24):
        """
        古いメトリクスをクリア

        Args:
            older_than_hours: 指定時間より古いメトリクスを削除
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

            # パフォーマンスメトリクスのクリア
            self._performance_metrics = deque(
                [m for m in self._performance_metrics if m.timestamp >= cutoff_time],
                maxlen=self.max_history,
            )

            # カスタムメトリクスのクリア
            for metric_name in list(self._metrics.keys()):
                self._metrics[metric_name] = deque(
                    [
                        m
                        for m in self._metrics[metric_name]
                        if m.timestamp >= cutoff_time
                    ],
                    maxlen=self.max_history,
                )


def performance_monitor(operation_name: str):
    """
    パフォーマンス監視デコレータ

    Args:
        operation_name: 操作名
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000

                # メトリクス記録
                metrics_collector.record_performance(
                    operation=operation_name,
                    duration_ms=duration_ms,
                    success=success,
                    error_message=error_message,
                )

        return wrapper

    return decorator


# グローバルメトリクス収集インスタンス
metrics_collector = MLMetricsCollector()


# 便利な関数エイリアス
def record_metric(name: str, value: float, **kwargs):
    """メトリクス記録"""
    metrics_collector.record_metric(name, value, **kwargs)


def record_performance(operation: str, duration_ms: float, **kwargs):
    """パフォーマンス記録"""
    metrics_collector.record_performance(operation, duration_ms, **kwargs)


def record_error(operation: str, error_type: str, error_message: str):
    """エラー記録"""
    metrics_collector.record_error(operation, error_type, error_message)


def get_metrics_summary(time_window_minutes: int = 60) -> Dict[str, Any]:
    """メトリクスサマリー取得"""
    return metrics_collector.get_metrics_summary(time_window_minutes)
