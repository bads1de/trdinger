"""
条件モデル

エントリー・イグジット条件の基本モデルを定義します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union


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
