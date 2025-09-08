"""
パフォーマンス測定ユーティリティ

パフォーマンス測定とログ出力に関連する共通機能を提供します。
"""

import logging

logger = logging.getLogger(__name__)


class PerformanceUtils:
    """パフォーマンス測定ユーティリティ"""

    @staticmethod
    def time_function(func):
        """関数実行時間測定デコレータ"""
        import time
        from functools import wraps
        from .logging_utils import LoggingUtils

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                LoggingUtils.log_performance(func.__name__, duration)
                return result
            except Exception:
                duration = time.time() - start_time
                LoggingUtils.log_performance(f"{func.__name__} (ERROR)", duration)
                raise

        return wrapper