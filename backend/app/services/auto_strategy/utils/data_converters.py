"""データ変換ユーティリティ"""

import logging
from typing import Any, Dict, List, Optional, Union

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
    def normalize_symbol(symbol: Optional[str], provider: str = "generic") -> str:
        """シンボルを正規化（統一サービス経由）"""
        if not symbol:
            return "BTC:USDT"
        return symbol.strip().upper()