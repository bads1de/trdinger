"""
取引基本設定

TradingSettings クラスを提供します。
"""

from dataclasses import dataclass, field
from typing import List

from app.config.constants import SUPPORTED_TIMEFRAMES

from .base import BaseConfig
from .constants import (
    CONSTRAINTS,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    SUPPORTED_SYMBOLS,
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
