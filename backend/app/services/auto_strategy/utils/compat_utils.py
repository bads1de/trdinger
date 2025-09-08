"""
互換性ユーティリティ関数

auto_strategy全体で使用される共通機能を提供します。
分割されたモジュールからの統一的なインポートを維持するための互換性ユーティリティ。
"""

import logging
from typing import Any, Optional
from app.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

# 互換性維持のための関数
safe_execute = ErrorHandler.safe_execute

def ensure_float(value: Any, default: float = 0.0) -> float:
    """値をfloatに安全に変換"""
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"float変換失敗: {value}, デフォルト値 {default} を使用")
        return default

def normalize_symbol(symbol: Optional[str], provider: str = "generic") -> str:
    """シンボルを正規化（統一サービス経由）"""
    if not symbol:
        return "BTC:USDT"
    return symbol.strip().upper()