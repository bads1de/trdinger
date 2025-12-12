"""
戦略遺伝子モデル
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .condition import Condition, ConditionGroup
from .entry_gene import EntryGene
from .enums import PositionSizingMethod
from .indicator_gene import IndicatorGene
from .position_sizing_gene import PositionSizingGene
from .stateful_condition import StatefulCondition
from .tpsl_gene import TPSLGene


@dataclass
class StrategyGene:
    """
    戦略遺伝子

    完全な取引戦略を表現する遺伝子です。
    """

    id: str = ""
    indicators: List[IndicatorGene] = field(default_factory=list)
    entry_conditions: List[Condition] = field(default_factory=list)
    exit_conditions: List[Condition] = field(default_factory=list)
    long_entry_conditions: List[Union[Condition, ConditionGroup]] = field(
        default_factory=list
    )
    short_entry_conditions: List[Union[Condition, ConditionGroup]] = field(
        default_factory=list
    )
    stateful_conditions: List[StatefulCondition] = field(default_factory=list)
    risk_management: Dict[str, Any] = field(default_factory=dict)
    tpsl_gene: Optional[TPSLGene] = None
    long_tpsl_gene: Optional[TPSLGene] = None
    short_tpsl_gene: Optional[TPSLGene] = None
    position_sizing_gene: Optional[PositionSizingGene] = None
    entry_gene: Optional[EntryGene] = None
    long_entry_gene: Optional[EntryGene] = None
    short_entry_gene: Optional[EntryGene] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_effective_long_conditions(self) -> List[Union[Condition, ConditionGroup]]:
        """有効なロング条件を取得（後方互換性を考慮）"""
        if self.long_entry_conditions:
            return self.long_entry_conditions
        elif self.entry_conditions:
            return list(self.entry_conditions)
        else:
            return []

    def get_effective_short_conditions(self) -> List[Union[Condition, ConditionGroup]]:
        """有効なショート条件を取得（後方互換性を考慮）"""
        if self.short_entry_conditions:
            return self.short_entry_conditions
        elif self.entry_conditions and not self.long_entry_conditions:
            return list(self.entry_conditions)
        else:
            return []

    def has_long_short_separation(self) -> bool:
        """ロング・ショート条件が分離されているかチェック"""
        return (
            self.long_entry_conditions is not None
            and len(self.long_entry_conditions) > 0
        ) or (
            self.short_entry_conditions is not None
            and len(self.short_entry_conditions) > 0
        )

    @property
    def method(self):
        """ポジションサイジングメソッドを取得（後方互換性のため）"""
        if self.position_sizing_gene and hasattr(self.position_sizing_gene, "method"):
            return self.position_sizing_gene.method
        else:
            return PositionSizingMethod.FIXED_RATIO

    def validate(self) -> Tuple[bool, List[str]]:
        """戦略遺伝子の妥当性を検証し、(is_valid, errors) を返す"""
        from .validator import GeneValidator

        validator = GeneValidator()
        is_valid, errors = validator.validate_strategy_gene(self)
        return is_valid, errors
