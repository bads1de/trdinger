"""
ML統一ログ機能

ML関連サービス共通のログ出力機能を提供します。
"""

import logging
import json
import time
from typing import Dict, Any, Optional, Callable, TypeVar
from functools import wraps
from datetime import datetime

from ..config import ml_config

T = TypeVar("T")


class MLStructuredLogger:
    """ML用構造化ログ出力クラス"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """ログ設定の初期化"""
        # ログレベルの設定
        log_level = getattr(
            logging, ml_config.data_processing.LOG_LEVEL.upper(), logging.INFO
        )
        self.logger.setLevel(log_level)

    def log_operation(
        self,
        operation: str,
        level: str = "INFO",
        duration_ms: Optional[float] = None,
        success: bool = True,
        **kwargs,
    ):
        """
        操作ログの出力

        Args:
            operation: 操作名
            level: ログレベル
            duration_ms: 処理時間（ミリ秒）
            success: 成功フラグ
            **kwargs: 追加のコンテキスト情報
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "success": success,
            **kwargs,
        }

        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms

        message = f"ML操作: {operation}"
        if duration_ms:
            message += f" ({duration_ms:.2f}ms)"

        # デバッグモードでは構造化ログも出力
        if ml_config.data_processing.DEBUG_MODE:
            message += f" | データ: {json.dumps(log_data, ensure_ascii=False)}"

        log_level = getattr(self.logger, level.lower())
        log_level(message)

    def log_performance(self, operation: str, metrics: Dict[str, float], **kwargs):
        """
        パフォーマンスメトリクスのログ出力

        Args:
            operation: 操作名
            metrics: パフォーマンスメトリクス
            **kwargs: 追加のコンテキスト情報
        """
        self.log_operation(
            operation=f"performance_{operation}",
            level="INFO",
            metrics=metrics,
            **kwargs,
        )

    def log_error(
        self,
        operation: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        エラーログの出力

        Args:
            operation: 操作名
            error: 発生したエラー
            context: エラーコンテキスト
            **kwargs: 追加の情報
        """
        error_data: Dict[str, Any] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if context:
            error_data["context"] = context

        self.log_operation(
            operation=f"error_{operation}",
            level="ERROR",
            success=False,
            **error_data,
            **kwargs,
        )

    def log_data_info(
        self,
        operation: str,
        data_shape: Optional[tuple] = None,
        data_size: Optional[int] = None,
        data_type: Optional[str] = None,
        **kwargs,
    ):
        """
        データ情報のログ出力

        Args:
            operation: 操作名
            data_shape: データの形状
            data_size: データサイズ
            data_type: データタイプ
            **kwargs: 追加の情報
        """
        data_info = {}
        if data_shape:
            data_info["data_shape"] = data_shape
        if data_size:
            data_info["data_size"] = data_size
        if data_type:
            data_info["data_type"] = data_type

        self.log_operation(
            operation=f"data_{operation}", level="DEBUG", **data_info, **kwargs
        )


def log_ml_operation(
    operation_name: str, log_performance: bool = True, log_data_info: bool = False
):
    """
    ML操作のログ出力デコレータ

    Args:
        operation_name: 操作名
        log_performance: パフォーマンスログを出力するか
        log_data_info: データ情報ログを出力するか
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            logger = MLStructuredLogger(func.__module__)

            start_time = time.time()

            try:
                # データ情報のログ出力
                if log_data_info and args:
                    first_arg = args[0]
                    if hasattr(first_arg, "shape"):
                        logger.log_data_info(
                            operation=operation_name,
                            data_shape=first_arg.shape,
                            data_size=(
                                first_arg.size if hasattr(first_arg, "size") else None
                            ),
                        )

                # 操作実行
                result = func(*args, **kwargs)

                # パフォーマンスログ出力
                if log_performance:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.log_operation(
                        operation=operation_name, duration_ms=duration_ms, success=True
                    )

                return result

            except Exception as e:
                # エラーログ出力
                duration_ms = (time.time() - start_time) * 1000
                logger.log_error(
                    operation=operation_name,
                    error=e,
                    context={"duration_ms": duration_ms},
                )
                raise

        return wrapper

    return decorator


def log_ml_metrics(
    operation_name: str,
    metrics: Dict[str, float],
    context: Optional[Dict[str, Any]] = None,
):
    """
    MLメトリクスのログ出力

    Args:
        operation_name: 操作名
        metrics: メトリクス
        context: コンテキスト情報
    """
    logger = MLStructuredLogger(__name__)
    logger.log_performance(operation=operation_name, metrics=metrics, **(context or {}))


def log_ml_data_summary(
    operation_name: str, data, additional_info: Optional[Dict[str, Any]] = None
):
    """
    MLデータサマリーのログ出力

    Args:
        operation_name: 操作名
        data: データ（pandas DataFrame, numpy array等）
        additional_info: 追加情報
    """
    logger = MLStructuredLogger(__name__)

    data_info = {}

    # データタイプ別の情報取得
    if hasattr(data, "shape"):
        data_info["shape"] = data.shape
    if hasattr(data, "size"):
        data_info["size"] = data.size
    if hasattr(data, "dtype"):
        data_info["dtype"] = str(data.dtype)
    if hasattr(data, "columns"):
        data_info["columns"] = list(data.columns)
    if hasattr(data, "index"):
        data_info["index_length"] = len(data.index)

    if additional_info:
        data_info.update(additional_info)

    logger.log_data_info(operation=operation_name, **data_info)


# グローバルロガーインスタンス
ml_logger = MLStructuredLogger("ml_services")


# 便利な関数エイリアス
def log_info(message: str, **kwargs):
    """情報ログの出力"""
    ml_logger.log_operation("info", level="INFO", message=message, **kwargs)


def log_warning(message: str, **kwargs):
    """警告ログの出力"""
    ml_logger.log_operation("warning", level="WARNING", message=message, **kwargs)


def log_error(message: str, error: Optional[Exception] = None, **kwargs):
    """エラーログの出力"""
    if error:
        ml_logger.log_error("error", error, context={"message": message}, **kwargs)
    else:
        ml_logger.log_operation("error", level="ERROR", message=message, **kwargs)


def log_debug(message: str, **kwargs):
    """デバッグログの出力"""
    ml_logger.log_operation("debug", level="DEBUG", message=message, **kwargs)
