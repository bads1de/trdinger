"""
ML統一ログ機能

ML関連サービス共通のログ出力機能を提供します。
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional, TypeVar

T = TypeVar("T")


class MLStructuredLogger:
    """ML用構造化ログ出力クラス"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        # 設定は遅延読み込みするためここでは初期化しない

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
        from app.config.unified_config import unified_config

        # ログレベルの設定（遅延適用）
        log_level_setting = getattr(
            logging, unified_config.ml.data_processing.log_level.upper(), logging.INFO
        )
        if self.logger.level != log_level_setting:
            self.logger.setLevel(log_level_setting)

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
        if unified_config.ml.data_processing.debug_mode:
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