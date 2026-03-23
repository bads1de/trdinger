"""
システム関連のユーティリティ関数
"""
import logging

logger = logging.getLogger(__name__)

def get_memory_usage_mb() -> float:
    """現在のプロセスのメモリ使用量を取得（MB単位）"""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0
    except Exception as e:
        logger.warning(f"メモリ使用量取得エラー: {e}")
        return 0.0
