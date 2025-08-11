from dataclasses import dataclass, field
from typing import List, Union

from .gene_strategy import Condition


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
        from .gene_validation import GeneValidator
        v = GeneValidator()
        for c in self.conditions:
            ok, _ = v.validate_condition(c)
            if not ok:
                return False
        return True

