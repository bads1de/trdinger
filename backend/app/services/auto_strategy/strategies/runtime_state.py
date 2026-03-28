"""
戦略実行時の可変状態を保持するモジュール。

UniversalStrategy 本体や helper が共有しているポジション状態を
明示的なオブジェクトに集約し、責務境界をはっきりさせる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


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


class LegacyStrategyRuntimeStateAdapter:
    """
    runtime_state を持たない既存 strategy オブジェクトを包む互換アダプタ。

    主に MagicMock ベースの既存ユニットテストを壊さずに manager 側を
    runtime_state ベースへ寄せるために使う。
    """

    __runtime_state_marker__: ClassVar[bool] = True

    def __init__(self, strategy: Any):
        self._strategy = strategy

    @property
    def sl_price(self) -> float | None:
        return getattr(self._strategy, "_sl_price", None)

    @sl_price.setter
    def sl_price(self, value: float | None) -> None:
        self._strategy._sl_price = value

    @property
    def tp_price(self) -> float | None:
        return getattr(self._strategy, "_tp_price", None)

    @tp_price.setter
    def tp_price(self, value: float | None) -> None:
        self._strategy._tp_price = value

    @property
    def entry_price(self) -> float | None:
        return getattr(self._strategy, "_entry_price", None)

    @entry_price.setter
    def entry_price(self, value: float | None) -> None:
        self._strategy._entry_price = value

    @property
    def position_direction(self) -> float:
        return getattr(self._strategy, "_position_direction", 0.0)

    @position_direction.setter
    def position_direction(self, value: float) -> None:
        self._strategy._position_direction = value

    @property
    def tp_reached(self) -> bool:
        return getattr(self._strategy, "_tp_reached", False)

    @tp_reached.setter
    def tp_reached(self, value: bool) -> None:
        self._strategy._tp_reached = value

    @property
    def trailing_tp_sl(self) -> float | None:
        return getattr(self._strategy, "_trailing_tp_sl", None)

    @trailing_tp_sl.setter
    def trailing_tp_sl(self, value: float | None) -> None:
        self._strategy._trailing_tp_sl = value

    def set_open_position(
        self,
        *,
        entry_price: float,
        sl_price: float | None,
        tp_price: float | None,
        direction: float,
    ) -> None:
        self.entry_price = entry_price
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.position_direction = direction
        self.tp_reached = False
        self.trailing_tp_sl = None

    def reset_position(self) -> None:
        self.sl_price = None
        self.tp_price = None
        self.entry_price = None
        self.position_direction = 0.0
        self.tp_reached = False
        self.trailing_tp_sl = None


def resolve_runtime_state(
    strategy: Any,
) -> StrategyRuntimeState | LegacyStrategyRuntimeStateAdapter:
    """strategy が持つ runtime_state を返し、なければ互換アダプタを返す。"""
    runtime_state = getattr(strategy, "runtime_state", None)
    if getattr(runtime_state, "__runtime_state_marker__", False) is True:
        return runtime_state  # type: ignore[return-value]
    return LegacyStrategyRuntimeStateAdapter(strategy)
