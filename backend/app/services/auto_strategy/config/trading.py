"""
TradingSettingsクラス

取引基本設定を管理します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import BaseConfig
from .constants import (
    CONSTRAINTS,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
)


@dataclass
class TradingSettings(BaseConfig):
    """取引基本設定"""

    # 基本取引設定
    default_symbol: str = DEFAULT_SYMBOL
    default_timeframe: str = DEFAULT_TIMEFRAME
    supported_symbols: List[str] = field(
        default_factory=lambda: SUPPORTED_SYMBOLS.copy()
    )
    supported_timeframes: List[str] = field(
        default_factory=lambda: SUPPORTED_TIMEFRAMES.copy()
    )

    # 運用制約
    min_trades: int = CONSTRAINTS["min_trades"]
    max_drawdown_limit: float = CONSTRAINTS["max_drawdown_limit"]
    max_position_size: float = CONSTRAINTS["max_position_size"]
    min_position_size: float = CONSTRAINTS["min_position_size"]

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得（自動生成を利用）"""
        # フィールドから自動生成したデフォルト値を取得
        defaults = self.get_default_values_from_fields()
        # 必要に応じてカスタマイズ（外部定数など）
        return defaults





