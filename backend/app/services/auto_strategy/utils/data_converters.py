"""データ変換ユーティリティ"""

import logging
import math
from typing import Any, Dict, Optional

from app.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

# 互換性維持用の関数
safe_execute = ErrorHandler.safe_execute


class DataConverter:
    """データ変換ユーティリティ"""

    @staticmethod
    def ensure_float(value: Any, default: float = 0.0) -> float:
        """値をfloatに安全に変換"""
        try:
            result = float(value)
            if not math.isfinite(result):
                logger.warning(
                    f"float変換失敗: {value} (無効値), デフォルト値 {default} を使用"
                )
                return default
            return result
        except (ValueError, TypeError, OverflowError):
            logger.warning(f"float変換失敗: {value}, デフォルト値 {default} を使用")
            return default

    @staticmethod
    def ensure_int(value: Any, default: int = 0) -> int:
        """値をintに安全に変換"""
        try:
            result = int(value)
            if not math.isfinite(result):
                logger.warning(
                    f"int変換失敗: {value} (無効値), デフォルト値 {default} を使用"
                )
                return default
            return result
        except (ValueError, TypeError, OverflowError):
            logger.warning(f"int変換失敗: {value}, デフォルト値 {default} を使用")
            return default

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
    def normalize_symbol(symbol: Optional[str], provider: str = "generic") -> str:
        """シンボルを正規化（統一サービス経由）"""
        if not symbol or not symbol.strip():
            return "BTC:USDT"
        return symbol.strip().upper()
