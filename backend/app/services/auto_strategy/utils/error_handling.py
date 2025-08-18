"""
統一エラーハンドリング（Auto Strategy専用）

ErrorHandlerを基盤として、Auto Strategy専用のエラーハンドリング機能を提供します。
既存のAutoStrategyErrorHandlerとcommon_utils.ErrorHandlerを統合し、一貫したエラー処理を実現します。
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, TypeVar


from app.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AutoStrategyErrorHandler(ErrorHandler):
    """
    Auto-Strategy用エラーハンドラークラス

    ErrorHandlerを継承し、Auto Strategy専用のエラーハンドリング機能を追加します。
    """

    # === Auto Strategy専用エラーハンドリング ===

    @staticmethod
    def handle_ga_error(error: Exception, context: str = "GA処理") -> Dict[str, Any]:
        """GA関連エラーの専用処理"""
        logger.error(f"GA処理エラー in {context}: {error}", exc_info=True)
        return AutoStrategyErrorHandler.create_error_response(
            message=str(error),
            error_code="GA_ERROR",
            context=context,
            details={"error_type": type(error).__name__},
        )

    @staticmethod
    def handle_strategy_generation_error(
        error: Exception,
        strategy_data: Optional[Dict] = None,
        context: str = "戦略生成",
    ) -> Dict[str, Any]:
        """戦略生成エラーの専用処理"""
        logger.error(
            f"戦略生成失敗 in {context}: {error}",
            extra={"strategy_data": strategy_data},
            exc_info=True,
        )
        return {
            "success": False,
            "strategy": None,
            "error": str(error),
            "context": context,
            "strategy_data": strategy_data,
        }

    @staticmethod
    def handle_calculation_error(
        error: Exception,
        context: str,
        fallback_value: Any = None,
        log_level: str = "error",
    ) -> Any:
        """
        計算エラーの標準的な処理（後方互換性のため保持）

        Args:
            error: 発生したエラー
            context: エラーのコンテキスト
            fallback_value: フォールバック値
            log_level: ログレベル

        Returns:
            フォールバック値
        """
        # ErrorHandlerの機能を活用
        return AutoStrategyErrorHandler.safe_execute(
            lambda: None,
            error_message=f"{context}でエラーが発生: {error}",
            default_return=fallback_value,
            log_level=log_level,
        )

    @staticmethod
    def handle_system_error(
        error: Exception, context: str, include_traceback: bool = True
    ) -> Dict[str, Any]:
        """
        システムエラーの標準的な処理

        Args:
            error: 発生したエラー
            context: エラーのコンテキスト
            include_traceback: トレースバックを含めるか

        Returns:
            エラー情報辞書
        """
        error_info = {
            "success": False,
            "error": str(error),
            "context": context,
            "error_type": type(error).__name__,
        }

        if include_traceback:
            error_info["traceback"] = traceback.format_exc()

        logger.error(f"{context}でシステムエラーが発生: {error}", exc_info=True)

        return error_info


class ErrorContext:
    """
    エラーコンテキスト管理クラス

    エラー発生時のコンテキスト情報を管理します。
    """

    def __init__(self, context: str):
        self.context = context
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def has_errors(self) -> bool:
        """エラーがあるかどうか"""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """警告があるかどうか"""
        return len(self.warnings) > 0

    def clear(self):
        """エラー・警告をクリア"""
        self.errors.clear()
        self.warnings.clear()
