"""
統一エラーハンドリング（Auto Strategy専用）

UnifiedErrorHandlerを基盤として、Auto Strategy専用のエラーハンドリング機能を提供します。
既存のAutoStrategyErrorHandlerとcommon_utils.ErrorHandlerを統合し、一貫したエラー処理を実現します。
"""

import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from functools import wraps

from app.utils.unified_error_handler import UnifiedErrorHandler

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AutoStrategyErrorHandler(UnifiedErrorHandler):
    """
    Auto-Strategy用エラーハンドラークラス

    UnifiedErrorHandlerを継承し、Auto Strategy専用のエラーハンドリング機能を追加します。
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
        # UnifiedErrorHandlerの機能を活用
        return AutoStrategyErrorHandler.safe_execute(
            lambda: None,
            error_message=f"{context}でエラーが発生: {error}",
            default_return=fallback_value,
            log_level=log_level,
        )

    @staticmethod
    def handle_validation_error(
        error: Exception, context: str, validation_errors: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        バリデーションエラーの標準的な処理

        Args:
            error: 発生したエラー
            context: エラーのコンテキスト
            validation_errors: 既存のバリデーションエラーリスト

        Returns:
            (False, error_messages)
        """
        error_msg = f"{context}でバリデーションエラーが発生: {error}"
        logger.error(error_msg, exc_info=True)

        errors = validation_errors or []
        errors.append(f"バリデーション処理エラー: {error}")

        return False, errors

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

    @staticmethod
    def safe_execute(
        func: Callable[..., T],
        *args,
        fallback_value: T = None,
        context: str = "関数実行",
        log_errors: bool = True,
        **kwargs,
    ) -> T:
        """
        安全な関数実行（後方互換性のため保持、UnifiedErrorHandlerに委譲）

        Args:
            func: 実行する関数
            *args: 関数の引数
            fallback_value: エラー時のフォールバック値
            context: エラーのコンテキスト
            log_errors: エラーをログ出力するか
            **kwargs: 関数のキーワード引数

        Returns:
            関数の実行結果またはフォールバック値
        """
        # 直接実装（無限再帰を避けるため）
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                logger.error(f"{context}でエラー: {e}", exc_info=True)
            return fallback_value

    @staticmethod
    def retry_on_failure(
        max_retries: int = 3,
        delay: float = 0.1,
        exceptions: Tuple[Exception, ...] = (Exception,),
    ):
        """
        失敗時のリトライデコレータ

        Args:
            max_retries: 最大リトライ回数
            delay: リトライ間隔（秒）
            exceptions: リトライ対象の例外タイプ
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                import time

                last_exception = None
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries:
                            logger.warning(
                                f"{func.__name__} 実行失敗 (試行 {attempt + 1}/{max_retries + 1}): {e}"
                            )
                            time.sleep(delay)
                        else:
                            logger.error(
                                f"{func.__name__} 最大リトライ回数に到達: {e}",
                                exc_info=True,
                            )

                raise last_exception

            return wrapper

        return decorator

    @staticmethod
    def validate_and_execute(
        validation_func: Callable[..., Tuple[bool, List[str]]],
        execution_func: Callable[..., T],
        validation_args: Tuple = (),
        execution_args: Tuple = (),
        validation_kwargs: Dict = None,
        execution_kwargs: Dict = None,
        context: str = "バリデーション付き実行",
    ) -> Tuple[bool, Union[T, List[str]]]:
        """
        バリデーション付きの安全な実行

        Args:
            validation_func: バリデーション関数
            execution_func: 実行関数
            validation_args: バリデーション関数の引数
            execution_args: 実行関数の引数
            validation_kwargs: バリデーション関数のキーワード引数
            execution_kwargs: 実行関数のキーワード引数
            context: エラーのコンテキスト

        Returns:
            (success, result_or_errors)
        """
        validation_kwargs = validation_kwargs or {}
        execution_kwargs = execution_kwargs or {}

        try:
            # バリデーション実行
            is_valid, errors = validation_func(*validation_args, **validation_kwargs)

            if not is_valid:
                return False, errors

            # 実行
            result = execution_func(*execution_args, **execution_kwargs)
            return True, result

        except Exception as e:
            error_info = AutoStrategyErrorHandler.handle_system_error(e, context)
            return False, [f"実行エラー: {error_info['error']}"]


class ErrorContext:
    """
    エラーコンテキスト管理クラス

    エラー発生時のコンテキスト情報を管理します。
    """

    def __init__(self, context: str):
        self.context = context
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, error: str):
        """エラーを追加"""
        self.errors.append(error)
        logger.error(f"[{self.context}] {error}")

    def add_warning(self, warning: str):
        """警告を追加"""
        self.warnings.append(warning)
        logger.warning(f"[{self.context}] {warning}")

    def has_errors(self) -> bool:
        """エラーがあるかどうか"""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """警告があるかどうか"""
        return len(self.warnings) > 0

    def get_summary(self) -> Dict[str, Any]:
        """エラー・警告のサマリーを取得"""
        return {
            "context": self.context,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "has_issues": self.has_errors() or self.has_warnings(),
        }

    def clear(self):
        """エラー・警告をクリア"""
        self.errors.clear()
        self.warnings.clear()


def error_boundary(context: str, fallback_value: Any = None, log_level: str = "error"):
    """
    エラーバウンダリデコレータ

    Args:
        context: エラーのコンテキスト
        fallback_value: エラー時のフォールバック値
        log_level: ログレベル
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return AutoStrategyErrorHandler.handle_calculation_error(
                    e, f"{context} ({func.__name__})", fallback_value, log_level
                )

        return wrapper

    return decorator
