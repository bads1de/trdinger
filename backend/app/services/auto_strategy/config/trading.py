"""
TradingSettingsクラス

取引基本設定を管理します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import BaseConfig
from ..constants import (
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    CONSTRAINTS,
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

    def _custom_validation(self) -> List[str]:
        """カスタム検証"""
        errors = []

        if self.default_symbol not in self.supported_symbols:
            errors.append(
                f"デフォルトシンボル '{self.default_symbol}' はサポート対象外です"
            )

        if self.default_timeframe not in self.supported_timeframes:
            errors.append(
                f"デフォルト時間軸 '{self.default_timeframe}' はサポート対象外です"
            )

        if self.min_position_size >= self.max_position_size:
            errors.append(
                "最小ポジションサイズは最大ポジションサイズより小さく設定してください"
            )

        return errors