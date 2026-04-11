"""
統一エラーハンドリング

APIErrorHandler と MLErrorHandler の重複機能を統合し、
一貫性のあるエラー処理とログ出力を提供します。
"""

import concurrent.futures
import functools
import inspect
import logging
import math
import platform
import signal
import time
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from fastapi import HTTPException, status

from .response import error_response

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutError(Exception):
    """統一タイムアウトエラー

    操作が指定された時間内に完了しなかった場合に送出されます。
    """


class ValidationError(Exception):
    """統一バリデーションエラー

    入力データの検証に失敗した場合に送出されます。
    """


class DataError(Exception):
    """統一データエラー

    データの読み込み、処理、保存中に問題が発生した場合に送出されます。
    """


class ModelError(Exception):
    """統一モデルエラー

    機械学習モデルのトレーニング、保存、読み込み、推論中に
    問題が発生した場合に送出されます。
    """


class ErrorHandler:
    """統一エラーハンドリングクラス

    API と ML 両方のコンテキストに対応した統一エラーハンドリング機能を提供します。
    APIErrorHandler と MLErrorHandler の重複機能を統合しています。

    主な機能:
    - 統一エラーレスポンスの生成
    - APIエラーのHTTPException変換
    - 安全な実行（safe_execute）デコレータ
    - 操作コンテキスト管理（operation_context）
    - タイムアウト処理
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
        error_message: str = "処理中にエラーが発生しました",
        log_level: str = "error",
        is_api_call: bool = False,
        api_status_code: int = 500,
        api_error_code: str = "INTERNAL_ERROR",
    ) -> Any:
        """
        関数を安全に実行し、例外を捕捉して適切に処理する

        Args:
            func: 実行する関数
            default_return: エラー時のデフォルト値
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

    @staticmethod
    def handle_timeout(
        func: Callable[..., T], timeout: int, *args: Any, **kwargs: Any
    ) -> T:
        """
        タイムアウト付きで関数を実行する（Windows/Unix両対応）

        Args:
            func: 実行する関数
            timeout: タイムアウト秒数
            *args: 関数に渡す位置引数
            **kwargs: 関数に渡すキーワード引数

        Returns:
            関数の実行結果

        Raises:
            TimeoutError: タイムアウト発生時
        """
        if platform.system() == "Windows":
            # Windows: スレッドベースのタイムアウト
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"操作がタイムアウトしました ({timeout}秒)")
        else:
            # Unix: シグナルベースのタイムアウト
            def timeout_handler(_signum: int, _frame: Any) -> None:
                raise TimeoutError(f"操作がタイムアウトしました ({timeout}秒)")

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)  # type: ignore[attr-defined]
            signal.alarm(timeout)  # type: ignore[attr-defined]
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)  # type: ignore[attr-defined]
                signal.signal(signal.SIGALRM, old_handler)  # type: ignore[attr-defined]

    @staticmethod
    def validate_predictions(predictions: Any) -> bool:
        """
        ML予測値をバリデーションする

        Args:
            predictions: 予測値（辞書形式、"up", "down", "range" キーが必要）

        Returns:
            バリデーション結果（True/False）
        """
        if predictions is None:
            logger.warning("予測値がNoneです")
            return False

        if not isinstance(predictions, dict):
            logger.warning(f"予測値は辞書形式である必要があります: {type(predictions)}")
            return False

        required_keys = {"up", "down", "range"}
        if not required_keys.issubset(predictions.keys()):
            missing = required_keys - set(predictions.keys())
            logger.warning(f"予測値に必須キーが不足しています: {missing}")
            return False

        for key, value in predictions.items():
            if value is None:
                logger.warning(f"予測値 '{key}' がNoneです")
                return False
            if isinstance(value, float):
                if math.isnan(value):
                    logger.warning(f"予測値 '{key}' にNaNが含まれています")
                    return False
                if math.isinf(value):
                    logger.warning(f"予測値 '{key}' にInfが含まれています")
                    return False
                if value < 0.0 or value > 1.0:
                    logger.warning(f"予測値 '{key}' が範囲外です: {value}")
                    return False

        total = sum(predictions[k] for k in required_keys)
        if total < 0.8 or total > 1.2:
            logger.warning(f"予測値の合計が範囲外です: {total}")
            return False

        return True

    @staticmethod
    def validate_dataframe(
        df: Any,
        required_columns: Optional[list] = None,
        min_rows: int = 0,
    ) -> bool:
        """
        DataFrameをバリデーションする

        Args:
            df: バリデーション対象のDataFrame
            required_columns: 必須カラムのリスト
            min_rows: 最小行数

        Returns:
            バリデーション結果（True/False）
        """
        import pandas as pd

        if df is None:
            logger.warning("データフレームが空です")
            return False

        if not isinstance(df, pd.DataFrame):
            logger.warning(f"DataFrame型である必要があります: {type(df)}")
            return False

        if df.empty:
            logger.warning("データフレームが空です")
            return False

        if min_rows > 0 and len(df) < min_rows:
            logger.warning(f"データ行数が不足しています: {len(df)} < {min_rows}")
            return False

        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"必須カラムが不足しています: {missing_columns}")
                return False

        return True


# --- デコレータとコンテキストマネージャー ---


def safe_operation(
    default_return: Any = "RAISE_EXCEPTION",
    error_handler: Optional[Callable] = None,
    context: str = "統一操作",
    is_api_call: bool = False,
):
    """
    関数（同期・非同期）を例外から保護し、一貫したエラーハンドリングを提供する共通デコレータです。

    このデコレータは、ラップされた関数内で例外が発生した場合に以下の順序で処理を試みます：
    1. `error_handler`（カスタム関数）が指定されている場合、それを呼び出して結果を返します。
    2. `is_api_call=True` の場合、FastAPI の標準化された `HTTPException` を送出します。
    3. `default_return` が "RAISE_EXCEPTION"（デフォルト）の場合、発生した例外をそのまま再送出します。
    4. それ以外の場合、エラーをログに記録し、`default_return` に指定された値を返却して実行を継続させます。

    Args:
        default_return (Any): エラー発生時に返却するデフォルト値。
        error_handler (Optional[Callable]): 独自のエラー処理ロジックを持つ関数 `(exception, context) -> Any`。
        context (str): ログ出力やエラーメッセージに使用される操作の識別名（例: "DB保存"）。
        is_api_call (bool): APIエンドポイントとして動作させるか（エラー時にHTTP例外を投げるか）。

    Returns:
        Callable: ラップされた関数。同期・非同期を自動判別します。
    """

    def _handle_error(e: Exception) -> Any:
        """エラーハンドリングの共通ロジック"""
        if error_handler:
            return error_handler(e, context)
        if is_api_call:
            raise ErrorHandler.handle_api_error(e, context)
        logger.error(f"エラー in {context}: {e}")
        if isinstance(default_return, str) and default_return == "RAISE_EXCEPTION":
            raise e
        return default_return

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    return _handle_error(e)

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return _handle_error(e)

            return sync_wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def operation_context(operation_name: str):
    """統一操作のコンテキストマネージャー

    操作の実行時間計測とログ出力を行うコンテキストマネージャーです。
    操作の開始・完了・エラーを自動的にログに記録します。

    Args:
        operation_name: 操作の識別名（ログメッセージに使用）。

    Yields:
        None: コンテキスト内での実行を許可する。

    Example:
        >>> with operation_context("データ処理"):
        ...     process_data()
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
    """現在のプロセスのメモリ使用量を取得する（MB単位）。

    psutilライブラリを使用して、現在実行中のPythonプロセスの
    RSS（Resident Set Size）メモリ使用量をMB単位で返します。

    Returns:
        float: メモリ使用量（MB）。psutilが利用できない場合や
            エラー発生時は0.0を返します。
    """
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
    """データベースが利用可能か確認し、初期化されていなければ500エラーを返す。

    APIエンドポイントの依存性注入として使用され、データベース接続と
    テーブル初期化を保証します。

    Args:
        error_message: エラー発生時に返すメッセージ。

    Raises:
        HTTPException: データベースの初期化に失敗した場合（500 Internal Server Error）。
    """
    from database.connection import init_db

    if init_db():
        return

    logger.error(error_message)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=error_message,
    )


def timeout_decorator(timeout_seconds: int = 30):
    """タイムアウトデコレータ

    関数の実行時間を制限し、指定時間内に完了しない場合は
    TimeoutErrorを送出します。

    Args:
        timeout_seconds: タイムアウト時間（秒）。デフォルト: 30秒。

    Returns:
        Callable: デコレートされた関数。タイムアウトした場合に
            TimeoutErrorを送出する。

    Raises:
        TimeoutError: 関数の実行がtimeout_secondsを超えた場合。

    Note:
        Windowsではスレッドベースのタイムアウト、Unixでは
        シグナルベースのタイムアウトを使用します。
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return ErrorHandler.handle_timeout(func, timeout_seconds, *args, **kwargs)

        return wrapper

    return decorator
