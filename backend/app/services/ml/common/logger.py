"""
ML統一ログ機能

ML関連サービス共通のログ出力機能を提供します。
"""

import json
import logging
from datetime import datetime

from typing import Dict, Optional, TypeVar

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


# グローバルロガーインスタンス
ml_logger = MLStructuredLogger("ml_services")


def log_error(message: str, error: Optional[Exception] = None, **kwargs):
    """エラーログの出力"""
    if error:
        ml_logger.log_error("error", error, context={"message": message}, **kwargs)
    else:
        ml_logger.log_operation("error", level="ERROR", message=message, **kwargs)


def log_info(message: str, **kwargs):
    """情報ログの出力"""
    ml_logger.log_operation("info", level="INFO", message=message, **kwargs)


def log_warning(message: str, **kwargs):
    """警告ログの出力"""
    ml_logger.log_operation("warning", level="WARNING", message=message, **kwargs)


def log_ml_operation(operation: str, **kwargs):
    """ML操作ログの出力"""
    ml_logger.log_operation(operation, **kwargs)


def log_ml_metrics(operation: str, metrics: Dict[str, float], **kwargs):
    """MLメトリクスログの出力"""
    ml_logger.log_performance(operation, metrics, **kwargs)


def log_ml_data_summary(operation: str, **kwargs):
    """MLデータサマリーログの出力"""
    ml_logger.log_operation(f"data_summary_{operation}", level="INFO", **kwargs)
