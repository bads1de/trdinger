"""
ML共通エラーハンドラー

ML関連サービス共通のエラーハンドリング機能を提供します。
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable, TypeVar
from functools import wraps
import pandas as pd
import numpy as np

from ..config import ml_config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MLCommonErrorHandler:
    """ML関連サービス共通のエラーハンドラー"""

    @staticmethod
    def handle_data_error(
        error: Exception, context: str, data_length: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        データエラーの統一処理

        Args:
            error: 発生したエラー
            context: エラーが発生したコンテキスト
            data_length: データ長（指定されない場合はデフォルト値を使用）

        Returns:
            デフォルトのML指標
        """
        logger.warning(f"データエラー in {context}: {error}")

        if data_length is None:
            data_length = ml_config.prediction.DEFAULT_INDICATOR_LENGTH

        return ml_config.prediction.get_default_indicators(data_length)

    @staticmethod
    def handle_prediction_error(error: Exception, context: str) -> Dict[str, float]:
        """
        予測エラーの統一処理

        Args:
            error: 発生したエラー
            context: エラーが発生したコンテキスト

        Returns:
            フォールバック予測値
        """
        logger.error(f"予測エラー in {context}: {error}")
        return ml_config.prediction.get_fallback_predictions()

    @staticmethod
    def handle_model_error(
        error: Exception, context: str, operation: str = "unknown"
    ) -> Dict[str, Any]:
        """
        モデルエラーの統一処理

        Args:
            error: 発生したエラー
            context: エラーが発生したコンテキスト
            operation: 実行していた操作

        Returns:
            エラー情報を含む辞書
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
    def handle_validation_error(
        error: Exception, context: str, data_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        バリデーションエラーの統一処理

        Args:
            error: 発生したエラー
            context: エラーが発生したコンテキスト
            data_info: データ情報（オプション）

        Returns:
            False（バリデーション失敗）
        """
        data_str = f" (データ情報: {data_info})" if data_info else ""
        logger.warning(f"バリデーションエラー in {context}: {error}{data_str}")
        return False

    @staticmethod
    def handle_timeout_error(
        error: Exception, context: str, timeout_seconds: float
    ) -> Dict[str, Any]:
        """
        タイムアウトエラーの統一処理

        Args:
            error: 発生したエラー
            context: エラーが発生したコンテキスト
            timeout_seconds: タイムアウト時間

        Returns:
            タイムアウト情報を含む辞書
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


def safe_ml_operation(
    default_return: Any = None,
    error_handler: Optional[Callable] = None,
    context: str = "ML操作",
):
    """
    ML操作を安全に実行するデコレータ

    Args:
        default_return: エラー時のデフォルト戻り値
        error_handler: カスタムエラーハンドラー
        context: 操作のコンテキスト
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
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


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
    min_rows: int = 1,
    context: str = "データフレーム検証",
) -> bool:
    """
    データフレームの統一バリデーション

    Args:
        df: 検証するデータフレーム
        required_columns: 必須カラムのリスト
        min_rows: 最小行数
        context: バリデーションのコンテキスト

    Returns:
        バリデーション結果
    """
    try:
        # 基本チェック
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

        # 必須カラムチェック
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(
                    f"{context}: 必須カラムが不足しています: {missing_columns}"
                )
                return False

        # 数値データチェック
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            logger.warning(f"{context}: 数値カラムが見つかりません")
            return False

        # NaN/Inf チェック
        if df.isnull().all().all():
            logger.warning(f"{context}: 全てのデータがNaNです")
            return False

        return True

    except Exception as e:
        return MLCommonErrorHandler.handle_validation_error(e, context)


def validate_ml_predictions(
    predictions: Dict[str, float], context: str = "ML予測値検証"
) -> bool:
    """
    ML予測値の統一バリデーション

    Args:
        predictions: 予測値の辞書
        context: バリデーションのコンテキスト

    Returns:
        バリデーション結果
    """
    try:
        return ml_config.prediction.validate_predictions(predictions)
    except Exception as e:
        return MLCommonErrorHandler.handle_validation_error(
            e, context, {"predictions": predictions}
        )


def validate_ml_indicators(
    indicators: Dict[str, np.ndarray],
    expected_length: Optional[int] = None,
    context: str = "ML指標検証",
) -> bool:
    """
    ML指標の統一バリデーション

    Args:
        indicators: ML指標の辞書
        expected_length: 期待される配列長
        context: バリデーションのコンテキスト

    Returns:
        バリデーション結果
    """
    try:
        required_indicators = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]

        # 必要な指標が存在するか
        if not all(indicator in indicators for indicator in required_indicators):
            missing = [ind for ind in required_indicators if ind not in indicators]
            logger.warning(f"{context}: 必要な指標が不足しています: {missing}")
            return False

        # 各指標の妥当性をチェック
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
        return MLCommonErrorHandler.handle_validation_error(
            e, context, {"indicators_keys": list(indicators.keys())}
        )


# 便利な関数エイリアス
handle_data_error = MLCommonErrorHandler.handle_data_error
handle_prediction_error = MLCommonErrorHandler.handle_prediction_error
handle_model_error = MLCommonErrorHandler.handle_model_error
handle_validation_error = MLCommonErrorHandler.handle_validation_error
handle_timeout_error = MLCommonErrorHandler.handle_timeout_error
