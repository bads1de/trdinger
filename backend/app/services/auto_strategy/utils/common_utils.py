"""
共通ユーティリティ関数

auto_strategy全体で使用される共通機能を提供します。
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from .error_handling import ErrorHandler

logger = logging.getLogger(__name__)


class DataConverter:
    """データ変換ユーティリティ"""

    @staticmethod
    def ensure_float(value: Any, default: float = 0.0) -> float:
        """値をfloatに安全に変換"""
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"float変換失敗: {value}, デフォルト値 {default} を使用")
            return default

    @staticmethod
    def ensure_int(value: Any, default: int = 0) -> int:
        """値をintに安全に変換"""
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"int変換失敗: {value}, デフォルト値 {default} を使用")
            return default

    @staticmethod
    def ensure_list(value: Any, default: Optional[List] = None) -> List:
        """値をリストに安全に変換"""
        if default is None:
            default = []

        if isinstance(value, list):
            return value
        elif value is None:
            return default
        else:
            return [value]

    @staticmethod
    def ensure_dict(value: Any, default: Optional[Dict] = None) -> Dict:
        """値を辞書に安全に変換"""
        if default is None:
            default = {}

        if isinstance(value, dict):
            return value
        else:
            return default

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """シンボルを正規化"""
        if not symbol:
            return "BTC:USDT"

        if ":" not in symbol:
            return f"{symbol}:USDT"

        return symbol



class LoggingUtils:
    """ログ出力ユーティリティ"""



    @staticmethod
    def log_performance(operation: str, duration: float, **metrics):
        """パフォーマンスログ"""
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        logger.info(f"[PERF] {operation}: {duration:.3f}s, {metrics_str}")



class ValidationUtils:
    """バリデーションユーティリティ"""

    @staticmethod
    def validate_range(
        value: Union[int, float],
        min_val: Union[int, float],
        max_val: Union[int, float],
        name: str = "値",
    ) -> bool:
        """範囲バリデーション"""
        if not (min_val <= value <= max_val):
            logger.warning(f"{name}が範囲外です: {value} (範囲: {min_val}-{max_val})")
            return False
        return True

    @staticmethod
    def validate_required_fields(
        data: Dict[str, Any], required_fields: List[str]
    ) -> tuple[bool, List[str]]:
        """必須フィールドバリデーション"""
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)

        if missing_fields:
            logger.warning(f"必須フィールドが不足しています: {missing_fields}")
            return False, missing_fields

        return True, []




class PerformanceUtils:
    """パフォーマンス測定ユーティリティ"""

    @staticmethod
    def time_function(func):
        """関数実行時間測定デコレータ"""
        import time
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                LoggingUtils.log_performance(func.__name__, duration)
                return result
            except Exception:
                duration = time.time() - start_time
                LoggingUtils.log_performance(f"{func.__name__} (ERROR)", duration)
                raise

        return wrapper



class CacheUtils:
    """キャッシュユーティリティ"""

    _cache = {}

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """キャッシュから値を取得"""
        return cls._cache.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """キャッシュに値を設定"""
        cls._cache[key] = {"value": value, "timestamp": datetime.now(), "ttl": ttl}

    @classmethod
    def clear(cls, pattern: Optional[str] = None) -> None:
        """キャッシュをクリア"""
        if pattern:
            keys_to_remove = [k for k in cls._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del cls._cache[key]
        else:
            cls._cache.clear()



# 便利な関数のエイリアス（AutoStrategyErrorHandlerに統合）
safe_execute = AutoStrategyErrorHandler.safe_execute
# safe_execute_async は ErrorHandler から利用可能
ensure_float = DataConverter.ensure_float
ensure_int = DataConverter.ensure_int
ensure_list = DataConverter.ensure_list
ensure_dict = DataConverter.ensure_dict
normalize_symbol = DataConverter.normalize_symbol
validate_range = ValidationUtils.validate_range
validate_required_fields = ValidationUtils.validate_required_fields
time_function = PerformanceUtils.time_function
