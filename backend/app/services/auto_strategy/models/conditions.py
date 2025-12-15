"""
条件モデル

エントリー・イグジット条件の基本モデルとステートフル条件を定義します。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ステートフル条件のエントリー方向
EntryDirection = Literal["long", "short"]


@dataclass
class Condition:
    """
    条件

    エントリー・イグジット条件を表現します。
    """

    left_operand: Union[Dict[str, Any], str, float]
    operator: str
    right_operand: Union[Dict[str, Any], str, float]

    def __post_init__(self) -> None:
        """型の正規化: 数値はfloatへ（テストの型要件に合わせる）"""
        try:
            if isinstance(self.left_operand, int):
                self.left_operand = float(self.left_operand)
            if isinstance(self.right_operand, int):
                self.right_operand = float(self.right_operand)
        except Exception:
            pass

    def validate(self) -> bool:
        """条件の妥当性を検証"""
        from .validator import GeneValidator

        validator = GeneValidator()
        return validator.validate_condition(self)[0]


@dataclass
class ConditionGroup:
    """
    条件グループ

    conditions の論理結合（operator）を表現します。
    再帰的な構造を持つことができます。
    """

    operator: str = "OR"
    conditions: List[Union[Condition, "ConditionGroup"]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.conditions) == 0

    def validate(self) -> bool:
        from .validator import GeneValidator

        validator = GeneValidator()

        # operator check
        if self.operator not in ["AND", "OR"]:
            return False

        for c in self.conditions:
            if isinstance(c, Condition):
                ok, _ = validator.validate_condition(c)
                if not ok:
                    return False
            elif isinstance(c, ConditionGroup):
                if not c.validate():
                    return False
            else:
                return False
        return True


class StateTracker:
    """
    状態追跡クラス

    バックテスト/ライブ実行中に発生したイベント（条件成立など）の
    バーインデックスを記録し、過去Nバー以内に発生したかを判定します。
    """

    def __init__(self):
        """初期化"""
        self._events: Dict[str, int] = {}  # event_name -> last_triggered_bar_index

    def record_event(self, event_name: str, bar_index: int) -> None:
        """
        イベントを記録

        Args:
            event_name: イベント名（一意の識別子）
            bar_index: イベントが発生したバーインデックス
        """
        self._events[event_name] = bar_index

    def was_triggered_within(
        self, event_name: str, lookback_bars: int, current_bar: int
    ) -> bool:
        """
        指定したイベントが過去Nバー以内に発生したかを判定

        Args:
            event_name: イベント名
            lookback_bars: 何バー以内を確認するか
            current_bar: 現在のバーインデックス

        Returns:
            過去lookback_bars以内にイベントが発生していればTrue
        """
        if event_name not in self._events:
            return False

        last_triggered = self._events[event_name]
        bars_since_trigger = current_bar - last_triggered

        return 0 <= bars_since_trigger <= lookback_bars

    def get_bars_since_event(self, event_name: str, current_bar: int) -> Optional[int]:
        """
        イベント発生からの経過バー数を取得

        Args:
            event_name: イベント名
            current_bar: 現在のバーインデックス

        Returns:
            経過バー数、イベントが未記録の場合はNone
        """
        if event_name not in self._events:
            return None
        return current_bar - self._events[event_name]

    def reset(self) -> None:
        """全イベントをクリア"""
        self._events.clear()

    def get_all_events(self) -> Dict[str, int]:
        """全イベントのコピーを取得（デバッグ用）"""
        return self._events.copy()


@dataclass
class StatefulCondition:
    """
    ステートフル条件

    シーケンシャルな条件を表現します:
    「trigger_condition が成立してから lookback_bars バー以内に
      follow_condition が成立した場合に True」

    Attributes:
        trigger_condition: トリガーとなる条件
        follow_condition: トリガー発生後に評価する条件
        lookback_bars: トリガー発生後、何バー以内にfollow_conditionが成立すれば有効か
        cooldown_bars: 条件成立後、次のトリガー記録までの待機バー数（オプション）
        direction: エントリー方向 ("long" または "short")。条件成立時の売買方向を指定。
        enabled: この条件が有効かどうか
    """

    trigger_condition: Condition
    follow_condition: Condition
    lookback_bars: int = 5
    cooldown_bars: int = 0
    direction: EntryDirection = "long"  # デフォルトはロング
    enabled: bool = True

    def validate(self) -> Tuple[bool, List[str]]:
        """
        バリデーション

        Returns:
            (is_valid, errors) のタプル
        """
        errors: List[str] = []

        # lookback_bars のチェック
        if self.lookback_bars <= 0:
            errors.append("lookback_bars must be greater than 0")

        # cooldown_bars のチェック
        if self.cooldown_bars < 0:
            errors.append("cooldown_bars must be non-negative")

        # trigger_condition のチェック
        if self.trigger_condition is None:
            errors.append("trigger_condition is required")
        elif not isinstance(self.trigger_condition, Condition):
            errors.append("trigger_condition must be a Condition instance")

        # follow_condition のチェック
        if self.follow_condition is None:
            errors.append("follow_condition is required")
        elif not isinstance(self.follow_condition, Condition):
            errors.append("follow_condition must be a Condition instance")

        return len(errors) == 0, errors

    def get_trigger_event_name(self) -> str:
        """
        一意のトリガーイベント名を生成

        trigger_condition の内容に基づいてハッシュを生成し、
        StateTracker で使用するイベント名として返します。

        Returns:
            一意のイベント名
        """
        # trigger_condition の内容をシリアライズしてハッシュを生成
        content = (
            f"{self.trigger_condition.left_operand}"
            f"{self.trigger_condition.operator}"
            f"{self.trigger_condition.right_operand}"
        )
        hash_value = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"stateful_trigger_{hash_value}"

    def __repr__(self) -> str:
        return (
            f"StatefulCondition("
            f"trigger={self.trigger_condition}, "
            f"follow={self.follow_condition}, "
            f"lookback={self.lookback_bars}, "
            f"cooldown={self.cooldown_bars})"
        )


