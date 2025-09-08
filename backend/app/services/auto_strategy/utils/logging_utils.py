"""
ログ出力ユーティリティ

パフォーマンス測定やログ出力のユーティリティを提供します。
"""

import logging

logger = logging.getLogger(__name__)


class LoggingUtils:
    """ログ出力ユーティリティ"""

    @staticmethod
    def log_performance(operation: str, duration: float, **metrics):
        """パフォーマンスログ"""
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        logger.info(f"[PERF] {operation}: {duration:.3f}s, {metrics_str}")