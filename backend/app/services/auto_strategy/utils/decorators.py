"""
Auto Strategy 共通デコレーター

safe_operationデコレーターの重複使用を統一し、
Auto Strategy固有の操作に対する統一デコレーターを提供します。
"""

import functools
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


def auto_strategy_operation(
    context: str = "Auto Strategy操作",
    default_return: Any = None,
    is_api_call: bool = False,
    log_level: str = "error",
    enable_debug_logging: bool = True,
):
    """
    Auto Strategy 統一操作デコレータ

    safe_operationの重複使用を統一し、Auto Strategy固有の
    エラーハンドリングとログ出力を提供します。

    Args:
        context: 操作コンテキスト（ログ出力時に使用）
        default_return: エラー発生時のデフォルト戻り値
        is_api_call: API呼び出しの場合はTrue
        log_level: ログレベル（'error', 'warning', 'info'）
        enable_debug_logging: デバッグログを有効にするかどうか

    Returns:
        デクレータ関数
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                result = func(*args, **kwargs)

                return result

            except Exception as e:
                # ログレベルの取得
                log_func = getattr(logger, log_level, logger.error)

                # 構造化されたエラーログ出力
                error_info = {
                    "operation": func.__name__,
                    "context": context,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }

                log_func(f"💥 {context}エラー: {error_info}")

                if is_api_call:
                    # API呼び出し時はHTTPExceptionを発生
                    from app.utils.error_handler import ErrorHandler

                    raise ErrorHandler.handle_api_error(
                        e,
                        context=context,
                        status_code=500,
                        error_code="AUTO_STRATEGY_ERROR",
                    )
                else:
                    # Auto Strategy操作のデフォルトはログ出力のみ
                    return default_return

        return wrapper

    return decorator


# 後方互換性のためのエイリアス
safe_auto_operation = auto_strategy_operation


def with_metrics_tracking(operation_name: str, track_memory: bool = False):
    """
    メトリクス追跡デコレータ（拡張用）

    Args:
        operation_name: 操作名
        track_memory: メモリ使用量も追跡する場合True

    Returns:
        デコレータ関数
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import time

            start_time = time.time()
            start_memory = None

            # Initialize psutil and os if memory tracking is enabled
            psutil_module = None
            os_module = None
            memory_tracking_enabled = track_memory

            if track_memory:
                try:
                    import psutil
                    import os

                    psutil_module = psutil
                    os_module = os
                    start_memory = psutil.Process(os.getpid()).memory_info().rss
                except ImportError:
                    logger.warning("psutil not available, memory tracking disabled")
                    memory_tracking_enabled = False

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time

                if (
                    memory_tracking_enabled
                    and start_memory is not None
                    and psutil_module
                    and os_module
                ):
                    end_memory = (
                        psutil_module.Process(os_module.getpid()).memory_info().rss
                    )
                    memory_usage = end_memory - start_memory
                    logger.info(
                        f"{operation_name} completed in {duration:.4f}s with memory change: {memory_usage / 1024:.1f} KB"
                    )
                else:
                    logger.info(f"{operation_name} completed in {duration:.4f}s")

        return wrapper

    return decorator
