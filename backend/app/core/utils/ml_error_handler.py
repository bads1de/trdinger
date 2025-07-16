"""
ML関連の統一エラーハンドリング

分散していたエラーハンドリング処理を統一し、
一貫性のあるエラー処理とログ出力を提供します。
"""

import logging
import platform
import signal
import concurrent.futures
import functools
import time
from typing import Any, Callable, Optional, Dict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MLTimeoutError(Exception):
    """ML処理のタイムアウトエラー"""

    pass


class MLValidationError(Exception):
    """ML関連のバリデーションエラー"""

    pass


class MLDataError(Exception):
    """MLデータ関連のエラー"""

    pass


class MLModelError(Exception):
    """MLモデル関連のエラー"""

    pass


class MLErrorHandler:
    """
    ML関連の統一エラーハンドリングクラス

    タイムアウト処理、エラーログ、デフォルト値の提供など、
    ML処理で共通的に必要なエラーハンドリング機能を提供します。
    """

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
            MLTimeoutError: タイムアウト時
        """
        try:
            if platform.system() == "Windows":
                return MLErrorHandler._handle_timeout_windows(
                    func, timeout_seconds, *args, **kwargs
                )
            else:
                return MLErrorHandler._handle_timeout_unix(
                    func, timeout_seconds, *args, **kwargs
                )
        except Exception as e:
            if "timeout" in str(e).lower():
                raise MLTimeoutError(
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
                raise MLTimeoutError(
                    f"Windows環境でのタイムアウト（{timeout_seconds}秒）"
                )

    @staticmethod
    def _handle_timeout_unix(
        func: Callable, timeout_seconds: int, *args, **kwargs
    ) -> Any:
        """Unix系環境でのタイムアウト処理"""

        def timeout_handler(signum, frame):
            raise MLTimeoutError(f"Unix環境でのタイムアウト（{timeout_seconds}秒）")

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

    @staticmethod
    def safe_execute(
        func: Callable,
        default_value: Any = None,
        error_message: str = "処理中にエラーが発生しました",
        log_level: str = "error",
    ) -> Any:
        """
        安全な関数実行（エラー時にデフォルト値を返す）

        Args:
            func: 実行する関数
            default_value: エラー時のデフォルト値
            error_message: エラーメッセージ
            log_level: ログレベル（error, warning, info）

        Returns:
            関数の実行結果またはデフォルト値
        """
        try:
            return func()
        except Exception as e:
            log_func = getattr(logger, log_level, logger.error)
            log_func(f"{error_message}: {e}")
            return default_value

    @staticmethod
    def validate_dataframe(
        df,
        required_columns: Optional[list] = None,
        min_rows: int = 1,
        allow_empty: bool = False,
    ) -> bool:
        """
        DataFrameの妥当性を検証

        Args:
            df: 検証するDataFrame
            required_columns: 必須カラムのリスト
            min_rows: 最小行数
            allow_empty: 空のDataFrameを許可するか

        Returns:
            妥当性検証の結果

        Raises:
            MLDataError: データが無効な場合
        """
        if df is None:
            if allow_empty:
                return True
            raise MLDataError("DataFrameがNoneです")

        if df.empty:
            if allow_empty:
                return True
            raise MLDataError("DataFrameが空です")

        if len(df) < min_rows:
            raise MLDataError(
                f"データ行数が不足しています（必要: {min_rows}行, 実際: {len(df)}行）"
            )

        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise MLDataError(f"必要なカラムが不足しています: {missing_columns}")

        return True

    @staticmethod
    def validate_predictions(predictions: Dict[str, float]) -> bool:
        """
        予測結果の妥当性を検証

        Args:
            predictions: 予測結果の辞書

        Returns:
            妥当性検証の結果

        Raises:
            MLValidationError: 予測結果が無効な場合
        """
        required_keys = ["up", "down", "range"]

        # 必要なキーの存在確認
        missing_keys = [key for key in required_keys if key not in predictions]
        if missing_keys:
            raise MLValidationError(
                f"予測結果に必要なキーが不足しています: {missing_keys}"
            )

        # 値の妥当性確認
        for key in required_keys:
            value = predictions[key]
            if not isinstance(value, (int, float)):
                raise MLValidationError(f"予測値が数値ではありません: {key}={value}")
            if not (0 <= value <= 1):
                raise MLValidationError(
                    f"予測値が範囲外です: {key}={value} (0-1の範囲である必要があります)"
                )

        # 合計値の確認
        total = sum(predictions[key] for key in required_keys)
        if not (0.8 <= total <= 1.2):
            raise MLValidationError(
                f"予測値の合計が範囲外です: {total} (0.8-1.2の範囲である必要があります)"
            )

        return True

    @staticmethod
    def get_default_predictions() -> Dict[str, float]:
        """デフォルトの予測値を取得"""
        return {"up": 0.33, "down": 0.33, "range": 0.34}

    @staticmethod
    def log_memory_usage(operation_name: str, threshold_mb: int = 1000):
        """メモリ使用量をログ出力"""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > threshold_mb:
                logger.warning(
                    f"{operation_name}: メモリ使用量が閾値を超過 ({memory_mb:.1f}MB > {threshold_mb}MB)"
                )
            else:
                logger.debug(f"{operation_name}: メモリ使用量 {memory_mb:.1f}MB")
        except ImportError:
            logger.debug(
                f"{operation_name}: psutilが利用できないため、メモリ監視をスキップ"
            )
        except Exception as e:
            logger.debug(f"{operation_name}: メモリ使用量の取得に失敗: {e}")


def timeout_decorator(timeout_seconds: int):
    """タイムアウト処理のデコレータ"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return MLErrorHandler.handle_timeout(func, timeout_seconds, *args, **kwargs)

        return wrapper

    return decorator


def safe_ml_operation(default_value=None, error_message="ML処理でエラーが発生しました"):
    """安全なML操作のデコレータ"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return MLErrorHandler.safe_execute(
                lambda: func(*args, **kwargs),
                default_value=default_value,
                error_message=error_message,
            )

        return wrapper

    return decorator


@contextmanager
def ml_operation_context(operation_name: str, log_memory: bool = True):
    """ML操作のコンテキストマネージャー"""
    start_time = time.time()

    if log_memory:
        MLErrorHandler.log_memory_usage(f"{operation_name} 開始")

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
            MLErrorHandler.log_memory_usage(f"{operation_name} 終了")
