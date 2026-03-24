"""
統一エラーハンドリング

APIErrorHandler と MLErrorHandler の重複機能を統合し、
一貫性のあるエラー処理とログ出力を提供します。
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from fastapi import HTTPException, status

from .response import error_response

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutError(Exception):
    """統一タイムアウトエラー"""


class ValidationError(Exception):
    """統一バリデーションエラー"""


class DataError(Exception):
    """統一データエラー"""


class ModelError(Exception):
    """統一モデルエラー"""


class ErrorHandler:
    """
    統一エラーハンドリングクラス

    API と ML 両方のコンテキストに対応した統一エラーハンドリング機能を提供します。
    APIErrorHandler と MLErrorHandler の重複機能を統合しています。
    """

    # --- 統一エラーレスポンス生成 ---

    @staticmethod
    def create_error_response(
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        統一エラーレスポンスを生成（response.make_error_responseへ委譲）

        Args:
            message: エラーメッセージ
            error_code: エラーコード
            details: エラー詳細
            context: エラーコンテキスト

        Returns:
            統一エラーレスポンス辞書
        """
        return error_response(
            message=message, error_code=error_code, details=details, context=context
        )

    # --- API エラーハンドリング ---

    @staticmethod
    def handle_api_error(
        error: Exception,
        context: str = "",
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
    ) -> HTTPException:
        """
        API エラーを処理し、HTTPException を生成

        Args:
            error: 発生したエラー
            context: エラーコンテキスト
            status_code: HTTPステータスコード
            error_code: エラーコード

        Returns:
            HTTPException
        """
        error_message = f"API エラー: {str(error)}"
        if context:
            error_message = f"{context} - {error_message}"

        logger.error(error_message, exc_info=True)

        return HTTPException(
            status_code=status_code,
            detail=ErrorHandler.create_error_response(
                message=str(error),
                error_code=error_code,
                context=context,
            ),
        )

    # --- ML エラーハンドリング ---

    @staticmethod
    def handle_model_error(
        error: Exception, context: str, operation: str = "unknown"
    ) -> Dict[str, Any]:
        """モデルエラーの統一処理"""
        logger.error(f"モデルエラー in {context} during {operation}: {error}")
        return ErrorHandler.create_error_response(
            message=str(error),
            error_code="MODEL_ERROR",
            context=context,
            details={
                "operation": operation,
                "error_type": type(error).__name__,
            },
        )

    # --- 統一安全実行 ---

    @staticmethod
    def safe_execute(
        func: Callable[..., Any],
        default_return: Optional[Any] = None,
        default_value: Optional[Any] = None,  # 後方互換性のため
        error_message: str = "処理中にエラーが発生しました",
        log_level: str = "error",
        is_api_call: bool = False,
        api_status_code: int = 500,
        api_error_code: str = "INTERNAL_ERROR",
    ) -> Any:
        """
        関数を安全に実行し、例外を捕捉して適切に処理する

        APIErrorHandler.handle_api_exception と MLErrorHandler.safe_execute の
        共通部分を統合した統一安全実行機能。

        Args:
            func: 実行する関数
            default_return: エラー時のデフォルト値
            default_value: エラー時のデフォルト値（後方互換性のため）
            error_message: エラーメッセージ
            log_level: ログレベル
            is_api_call: API呼び出しかどうか
            api_status_code: APIエラー時のステータスコード
            api_error_code: APIエラーコード

        Returns:
            関数の実行結果またはデフォルト値

        Raises:
            HTTPException: API呼び出し時のエラー
        """
        # 後方互換性の処理
        if default_value is not None and default_return is None:
            default_return = default_value
        try:
            return func()
        except HTTPException as e:
            if is_api_call:
                # 既にHTTPExceptionの場合はそのまま再raise
                logger.error(
                    f"API例外処理: {error_message} - {e.detail}", exc_info=True
                )
                raise e
            else:
                # MLコンテキストでHTTPExceptionが発生した場合の処理
                logger.error(f"ML処理中にAPI例外が発生: {e.detail}")
                return default_return
        except Exception as e:
            if is_api_call:
                # API関連のエラーとして処理
                raise ErrorHandler.handle_api_error(
                    e,
                    context=error_message,
                    status_code=api_status_code,
                    error_code=api_error_code,
                )
            else:
                # ML関連のエラーとして処理
                log_func = getattr(logger, log_level, logger.error)
                log_func(f"{error_message}: {e}")
                return default_return

    @staticmethod
    async def safe_execute_async(
        call: Callable[..., Awaitable[Any]],
        message: str = "Internal Server Error",
        status_code: int = 500,
    ) -> Any:
        """
        非同期関数を安全に実行（API用）

        Args:
            call: 実行する非同期関数
            message: エラーメッセージ
            status_code: エラー時のステータスコード

        Returns:
            関数の実行結果

        Raises:
            HTTPException: エラー時
        """
        try:
            return await call()
        except HTTPException as e:
            logger.error(f"API例外処理: {message} - {e.detail}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"API例外処理: {message} - {e}", exc_info=True)
            raise HTTPException(status_code=status_code, detail=message)

    @staticmethod
    def api_safe_execute(
        message: str = "Internal Server Error",
        status_code: int = 500,
    ):
        """
        APIエンドポイント用の安全な実行デコレータ

        Args:
            message: エラーメッセージ
            status_code: エラー時のHTTPステータスコード
        """
        def decorator(func: Callable[..., Awaitable[Any]]):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except HTTPException as e:
                    logger.error(f"API例外処理: {message} - {e.detail}", exc_info=True)
                    raise e
                except Exception as e:
                    logger.error(f"API例外処理: {message} - {e}", exc_info=True)
                    raise HTTPException(status_code=status_code, detail=message)
            return wrapper
        return decorator

# --- デコレータとコンテキストマネージャー ---


def safe_operation(
    default_return: Any = "RAISE_EXCEPTION",
    error_handler: Optional[Callable] = None,
    context: str = "統一操作",
    is_api_call: bool = False,
):
    """
    統一安全操作デコレータ

    API と ML 両方のコンテキストに対応した安全実行デコレータ
    default_returnが指定されていない場合や"RAISE_EXCEPTION"の場合、例外を投げる
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    return error_handler(e, context)
                else:
                    if is_api_call:
                        raise ErrorHandler.handle_api_error(e, context)
                    else:
                        logger.error(f"エラー in {context}: {e}")
                        if (
                            isinstance(default_return, str)
                            and default_return == "RAISE_EXCEPTION"
                        ):
                            raise e
                        return default_return

        return wrapper

    return decorator


@contextmanager
def operation_context(operation_name: str):
    """
    統一操作のコンテキストマネージャー

    操作の実行時間計測を行うコンテキストマネージャー

    Args:
        operation_name: 操作名
    """
    start_time = time.time()

    logger.info(f"{operation_name} を開始")

    try:
        yield
        duration = time.time() - start_time
        logger.info(f"{operation_name} が完了しました（{duration:.2f}秒）")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"{operation_name} でエラーが発生しました（{duration:.2f}秒）: {e}"
        )
        raise


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


# 標準的なエイリアス
safe_execute = ErrorHandler.safe_execute
api_safe_execute = ErrorHandler.api_safe_execute
safe_ml_operation = safe_operation
ml_operation_context = operation_context
 
# API用 DB初期化確認ヘルパー
DEFAULT_DB_INIT_ERROR_MESSAGE = "データベースの初期化に失敗しました"
 
 
def ensure_db_initialized(
    error_message: str = DEFAULT_DB_INIT_ERROR_MESSAGE,
) -> None:
    """DB が利用可能でなければ 500 を返す。"""
    from database.connection import init_db
 
    if init_db():
        return
 
    logger.error(error_message)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=error_message,
    )



