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
import traceback
from typing import Any, Callable, Optional, Dict, TypeVar
from contextlib import contextmanager
import pandas as pd
import numpy as np

from app.core.services.ml.config import ml_config

logger = logging.getLogger(__name__)

T = TypeVar("T")


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

    # --- エラーハンドリングメソッド ---

    @staticmethod
    def handle_data_error(
        error: Exception, context: str, data_length: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        データエラーの統一処理
        """
        logger.warning(f"データエラー in {context}: {error}")
        if data_length is None:
            data_length = ml_config.prediction.DEFAULT_INDICATOR_LENGTH
        return ml_config.prediction.get_default_indicators(data_length)

    @staticmethod
    def handle_prediction_error(error: Exception, context: str) -> Dict[str, float]:
        """
        予測エラーの統一処理
        """
        logger.error(f"予測エラー in {context}: {error}")
        return ml_config.prediction.get_fallback_predictions()

    @staticmethod
    def handle_model_error(
        error: Exception, context: str, operation: str = "unknown"
    ) -> Dict[str, Any]:
        """
        モデルエラーの統一処理
        """
        logger.error(f"モデルエラー in {context} during {operation}: {error}")
        return {
            "success": False,
            "error": str(error),
            "context": context,
            "operation": operation,
            "error_type": type(error).__name__,
        }

    @staticmethod
    def handle_timeout_error(
        error: Exception, context: str, timeout_seconds: float
    ) -> Dict[str, Any]:
        """
        タイムアウトエラーの統一処理
        """
        logger.error(
            f"タイムアウトエラー in {context} after {timeout_seconds}s: {error}"
        )
        return {
            "success": False,
            "error": "timeout",
            "timeout_seconds": timeout_seconds,
            "context": context,
            "message": f"処理が{timeout_seconds}秒でタイムアウトしました",
        }

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
        df: pd.DataFrame,
        required_columns: Optional[list] = None,
        min_rows: int = 1,
        context: str = "データフレーム検証",
    ) -> bool:
        """
        データフレームの統一バリデーション
        """
        try:
            if df is None:
                logger.warning(f"{context}: データフレームがNoneです")
                return False
            if df.empty:
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
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                logger.warning(f"{context}: 数値カラムが見つかりません")
                return False
            if df.isnull().all().all():
                logger.warning(f"{context}: 全てのデータがNaNです")
                return False
            return True
        except Exception as e:
            MLErrorHandler.handle_model_error(
                e, context, operation="validate_dataframe"
            )
            return False

    @staticmethod
    def validate_predictions(
        predictions: Dict[str, float], context: str = "ML予測値検証"
    ) -> bool:
        """
        ML予測値の統一バリデーション
        """
        try:
            return ml_config.prediction.validate_predictions(predictions)
        except Exception as e:
            MLErrorHandler.handle_model_error(
                e, context, operation="validate_predictions"
            )
            return False

    @staticmethod
    def get_default_predictions() -> Dict[str, float]:
        """デフォルトの予測値を取得"""
        return ml_config.prediction.get_fallback_predictions()

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

    @staticmethod
    def validate_ml_indicators(
        indicators: Dict[str, np.ndarray],
        expected_length: Optional[int] = None,
        context: str = "ML指標検証",
    ) -> bool:
        """
        ML指標の統一バリデーション
        """
        try:
            required_indicators = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
            if not all(indicator in indicators for indicator in required_indicators):
                missing = [ind for ind in required_indicators if ind not in indicators]
                logger.warning(f"{context}: 必要な指標が不足しています: {missing}")
                return False
            for indicator, values in indicators.items():
                if not isinstance(values, np.ndarray):
                    logger.warning(f"{context}: {indicator}が配列ではありません")
                    return False
                if len(values) == 0:
                    logger.warning(f"{context}: {indicator}が空の配列です")
                    return False
                if not np.all((values >= 0) & (values <= 1)):
                    logger.warning(f"{context}: {indicator}の値が0-1範囲外です")
                    return False
                if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                    logger.warning(f"{context}: {indicator}に無効な値が含まれています")
                    return False
                if expected_length and len(values) != expected_length:
                    logger.warning(
                        f"{context}: {indicator}の長さが期待値と異なります ({len(values)} != {expected_length})"
                    )
                    return False
            return True
        except Exception as e:
            MLErrorHandler.handle_model_error(
                e, context, operation="validate_ml_indicators"
            )
            return False


def timeout_decorator(timeout_seconds: int):
    """タイムアウト処理のデコレータ"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return MLErrorHandler.handle_timeout(func, timeout_seconds, *args, **kwargs)

        return wrapper

    return decorator


def safe_ml_operation(
    default_return: Any = None,
    error_handler: Optional[Callable] = None,
    context: str = "ML操作",
):
    """
    ML操作を安全に実行するデコレータ
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
                    logger.error(f"エラー in {context}: {e}")
                    if ml_config.data_processing.DEBUG_MODE:
                        logger.debug(f"スタックトレース: {traceback.format_exc()}")
                    return default_return

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
