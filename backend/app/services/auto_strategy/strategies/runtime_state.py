"""
戦略実行時の可変状態を保持するモジュール。

UniversalStrategy 本体や helper が共有しているポジション状態を
明示的なオブジェクトに集約し、責務境界をはっきりさせる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class StrategyRuntimeState:
    """戦略の実行時に変化するポジション状態。"""

    __runtime_state_marker__: ClassVar[bool] = True

    sl_price: float | None = None
    tp_price: float | None = None
    entry_price: float | None = None
    position_direction: float = 0.0
    tp_reached: bool = False
    trailing_tp_sl: float | None = None

    def set_open_position(
        self,
        *,
        entry_price: float,
        sl_price: float | None,
        tp_price: float | None,
        direction: float,
    ) -> None:
        """新規ポジションの状態を反映する。"""
        self.entry_price = entry_price
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.position_direction = direction
        self.tp_reached = False
        self.trailing_tp_sl = None

    def reset_position(self) -> None:
        """ポジション決済後に状態を初期化する。"""
        self.sl_price = None
        self.tp_price = None
        self.entry_price = None
        self.position_direction = 0.0
        self.tp_reached = False
        self.trailing_tp_sl = None
