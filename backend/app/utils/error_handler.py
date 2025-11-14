"""
統一エラーハンドリング

APIErrorHandler と MLErrorHandler の重複機能を統合し、
一貫性のあるエラー処理とログ出力を提供します。
"""

import concurrent.futures
import functools
import logging
import platform
import signal
import time
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

import numpy as np
import pandas as pd
from fastapi import HTTPException

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
        # delegate to shared response util to ensure consistent format
        from .response import error_response

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

    # --- タイムアウト処理 ---

    @staticmethod
    def handle_timeout(func: Callable, timeout_seconds: int, *args, **kwargs) -> Any:
        """
        プラットフォーム対応のタイムアウト処理

        Args:
            func: 実行する関数
            timeout_seconds: タイムアウト時間（秒）
            *args: 関数の引数
            **kwargs: 関数のキーワード引数

        Returns:
            関数の実行結果

        Raises:
            TimeoutError: タイムアウト時
        """
        try:
            if platform.system() == "Windows":
                return ErrorHandler._handle_timeout_windows(
                    func, timeout_seconds, *args, **kwargs
                )
            else:
                return ErrorHandler._handle_timeout_unix(
                    func, timeout_seconds, *args, **kwargs
                )
        except Exception as e:
            if "timeout" in str(e).lower():
                raise TimeoutError(
                    f"処理がタイムアウトしました（{timeout_seconds}秒）: {e}"
                )
            raise

    @staticmethod
    def _handle_timeout_windows(
        func: Callable, timeout_seconds: int, *args, **kwargs
    ) -> Any:
        """Windows環境でのタイムアウト処理"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Windows環境でのタイムアウト（{timeout_seconds}秒）"
                )

    @staticmethod
    def _handle_timeout_unix(
        func: Callable, timeout_seconds: int, *args, **kwargs
    ) -> Any:
        """Unix系環境でのタイムアウト処理"""

        def timeout_handler(signum, frame):
            _ = signum, frame  # 未使用パラメータ
            raise TimeoutError(f"Unix環境でのタイムアウト（{timeout_seconds}秒）")

        # 既存のシグナルハンドラーを保存
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # タイムアウトをクリア
            return result
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)  # 元のハンドラーを復元

    # --- バリデーション機能 ---

    @staticmethod
    def validate_predictions(
        predictions: Dict[str, float], context: str = "統一予測値検証"
    ) -> bool:
        """
        予測値の統一バリデーション

        Note: 実装は MLConfigValidator.validate_predictions を使用

        Args:
            predictions: 予測値の辞書
            context: バリデーションコンテキスト

        Returns:
            バリデーション結果
        """
        try:
            from app.config.validators import MLConfigValidator

            # 基本的なバリデーションを実行
            is_valid = MLConfigValidator.validate_predictions(predictions)

            if not is_valid:
                logger.warning(f"{context}: 予測値のバリデーションに失敗しました")

            # 追加のNaN/Inf チェック
            if is_valid and predictions is not None:
                for key, value in predictions.items():
                    if np.isnan(value) or np.isinf(value):
                        logger.warning(f"{context}: {key}に無効な値が含まれています")
                        return False

            return is_valid

        except Exception as e:
            ErrorHandler.handle_model_error(
                e, context, operation="validate_predictions"
            )
            return False

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: Optional[list] = None,
        min_rows: int = 1,
        context: str = "統一データフレーム検証",
    ) -> bool:
        """
        データフレームの統一バリデーション

        Args:
            df: 検証するデータフレーム
            required_columns: 必須カラムのリスト
            min_rows: 最小行数
            context: バリデーションコンテキスト

        Returns:
            バリデーション結果
        """
        try:
            if df is None or df.empty:
                logger.warning(f"{context}: データフレームが空です")
                return False

            if len(df) < min_rows:
                logger.warning(
                    f"{context}: データ行数が不足しています ({len(df)} < {min_rows})"
                )
                return False

            if required_columns:
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]
                if missing_columns:
                    logger.warning(
                        f"{context}: 必須カラムが不足しています: {missing_columns}"
                    )
                    return False

            return True
        except Exception as e:
            ErrorHandler.handle_model_error(e, context, operation="validate_dataframe")
            return False


# --- デコレータとコンテキストマネージャー ---


def timeout_decorator(timeout_seconds: int):
    """
    タイムアウトデコレータ

    指定された秒数で関数の実行をタイムアウトさせるデコレータ

    Args:
        timeout_seconds: タイムアウト秒数

    Returns:
        デコレータ関数
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return ErrorHandler.handle_timeout(func, timeout_seconds, *args, **kwargs)

        return wrapper

    return decorator


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
def operation_context(operation_name: str, log_memory: bool = False):
    """
    統一操作のコンテキストマネージャー

    操作の実行時間とメモリ使用量を計測するコンテキストマネージャー

    Args:
        operation_name: 操作名
        log_memory: メモリ使用量を記録するかどうか
    """
    start_time = time.time()

    if log_memory:
        # メモリ使用量のログ（必要に応じて実装）
        pass

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
    finally:
        if log_memory:
            # メモリ使用量のログ（必要に応じて実装）
            pass


# 標準的なエイリアス
handle_api_exception = ErrorHandler.safe_execute_async
safe_execute = ErrorHandler.safe_execute
safe_ml_operation = safe_operation
ml_operation_context = operation_context
