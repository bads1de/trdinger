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
    OR条件グループ

    conditions のいずれかが True なら True。
    ANDで使う側の配列にこのグループを1要素として混在させることで、
    A AND (B OR C) を表現できる。
    """

    conditions: List[Condition] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.conditions) == 0

    def validate(self) -> bool:
        from .validator import GeneValidator

        validator = GeneValidator()
        for c in self.conditions:
            ok, _ = validator.validate_condition(c)
            if not ok:
                return False
        return True
