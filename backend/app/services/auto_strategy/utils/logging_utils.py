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
        # durationがNoneでないことを確認
        if duration is None or not isinstance(duration, (int, float)):
            logger.warning(f"[PERF] Invalid duration for {operation}: {duration}")
            return

        try:
            # メトリクスをソートして順序保証
            sorted_metrics = sorted(metrics.items(), key=lambda x: x[0])
            metrics_str = ", ".join([f"{k}={v}" for k, v in sorted_metrics])
            logger.info(f"[PERF] {operation}: {duration:.3f}s, {metrics_str}")
        except Exception as e:
            logger.warning(f"[PERF] Error logging performance for {operation}: {e}")