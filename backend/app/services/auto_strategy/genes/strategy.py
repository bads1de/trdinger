"""
戦略遺伝子モデル

GA（遺伝的アルゴリズム）によって進化させる取引戦略の設計図を表します。
インジケーター、エントリー条件、エグジット条件、リスク管理などの
遺伝子を統合し、交叉・突然変異・クローン操作をサポートします。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

from .conditions import Condition, ConditionGroup, StatefulCondition
from .entry import EntryGene
from .indicator import IndicatorGene
from .position_sizing import PositionSizingGene
from .tool import ToolGene
from .tpsl import TPSLGene

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StrategyGene:
    """
    戦略遺伝子

    完全な取引戦略を表現する遺伝子です。
    """

    id: str = ""
    indicators: List[IndicatorGene] = field(default_factory=list)
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
    tool_genes: List[ToolGene] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_default(cls) -> "StrategyGene":
        """デフォルトの戦略遺伝子を作成。"""
        from .strategy_factory import create_default_strategy_gene

        return create_default_strategy_gene(cls)

    @classmethod
    def assemble(
        cls,
        indicators: List[IndicatorGene],
        long_entry_conditions: List[Union[Condition, ConditionGroup]],
        short_entry_conditions: List[Union[Condition, ConditionGroup]],
        tpsl_gene: Optional[TPSLGene] = None,
        position_sizing_gene: Optional[PositionSizingGene] = None,
        long_tpsl_gene: Optional[TPSLGene] = None,
        short_tpsl_gene: Optional[TPSLGene] = None,
        entry_gene: Optional[EntryGene] = None,
        long_entry_gene: Optional[EntryGene] = None,
        short_entry_gene: Optional[EntryGene] = None,
        tool_genes: Optional[List[ToolGene]] = None,
        risk_management: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "StrategyGene":
        """パーツから戦略遺伝子を組み立てる。"""
        from .strategy_factory import assemble_strategy_gene

        return assemble_strategy_gene(
            cls,
            indicators=indicators,
            long_entry_conditions=long_entry_conditions,
            short_entry_conditions=short_entry_conditions,
            tpsl_gene=tpsl_gene,
            position_sizing_gene=position_sizing_gene,
            long_tpsl_gene=long_tpsl_gene,
            short_tpsl_gene=short_tpsl_gene,
            entry_gene=entry_gene,
            long_entry_gene=long_entry_gene,
            short_entry_gene=short_entry_gene,
            tool_genes=tool_genes,
            risk_management=risk_management,
            metadata=metadata,
        )

    def has_long_short_separation(self) -> bool:
        """ロング・ショート条件が分離されているかチェック（常にTrue）。"""
        return True

    def validate(self) -> Tuple[bool, List[str]]:
        """戦略遺伝子の妥当性を検証し、(is_valid, errors) を返す。"""
        from .validator import GeneValidator

        validator = GeneValidator()
        is_valid, errors = validator.validate_strategy_gene(self)
        return is_valid, errors

    @classmethod
    def clone_field_names(cls) -> Tuple[str, ...]:
        """clone 対象となるフィールド名を返す。"""
        return tuple(
            field_info.name for field_info in fields(cls) if field_info.name != "id"
        )

    @classmethod
    def crossover_field_names(cls) -> Tuple[str, ...]:
        """uniform crossover 対象となるフィールド名を返す。"""
        return tuple(name for name in cls.clone_field_names() if name != "metadata")

    def clone(self, keep_id: bool = False) -> "StrategyGene":
        """軽量コピーを作成。"""
        from .genetic_utils import GeneticUtils

        cloned_fields = {
            field_name: GeneticUtils.smart_copy(getattr(self, field_name))
            for field_name in self.clone_field_names()
        }
        cloned_fields["id"] = self.id if keep_id else str(uuid.uuid4())
        return type(self)(**cloned_fields)

    def mutate(self, config: Any, mutation_rate: float = 0.1) -> "StrategyGene":
        """戦略遺伝子の突然変異を実行する。"""
        from .strategy_operators import mutate_strategy_gene

        return mutate_strategy_gene(self, config, mutation_rate=mutation_rate)

    def adaptive_mutate(
        self, population: List[Any], config: Any, base_mutation_rate: float = 0.1
    ) -> "StrategyGene":
        """集団の多様性に基づいて変異率を調整する。"""
        from .strategy_operators import adaptive_mutate_strategy_gene

        return adaptive_mutate_strategy_gene(
            self,
            population,
            config,
            base_mutation_rate=base_mutation_rate,
        )

    @classmethod
    def crossover(
        cls,
        parent1: "StrategyGene",
        parent2: "StrategyGene",
        config: Any,
        crossover_type: str = "uniform",
    ) -> Tuple["StrategyGene", "StrategyGene"]:
        """2つの親個体から交叉により新しい子個体を生成する。"""
        from .strategy_operators import crossover_strategy_genes

        return crossover_strategy_genes(
            cls,
            parent1,
            parent2,
            config,
            crossover_type=crossover_type,
        )

    @staticmethod
    def _mutate_indicators(
        mutated: "StrategyGene", mutation_rate: float, config: Any
    ) -> None:
        """指標遺伝子の突然変異処理。"""
        from .strategy_operators import mutate_indicators

        mutate_indicators(mutated, mutation_rate, config)

    @staticmethod
    def _mutate_conditions(
        mutated: "StrategyGene", mutation_rate: float, config: Any
    ) -> None:
        """取引条件の突然変異処理。"""
        from .strategy_operators import mutate_conditions

        mutate_conditions(mutated, mutation_rate, config)

    @staticmethod
    def _crossover_tpsl_genes(
        parent1_tpsl: Optional[TPSLGene],
        parent2_tpsl: Optional[TPSLGene],
    ) -> Tuple[Optional[TPSLGene], Optional[TPSLGene]]:
        """TPSL 遺伝子の交叉処理。"""
        from .strategy_operators import crossover_tpsl_genes

        return crossover_tpsl_genes(parent1_tpsl, parent2_tpsl)

    @staticmethod
    def _crossover_position_sizing_genes(
        parent1_ps: Optional[PositionSizingGene],
        parent2_ps: Optional[PositionSizingGene],
    ) -> Tuple[Optional[PositionSizingGene], Optional[PositionSizingGene]]:
        """ポジションサイズ遺伝子の交叉処理。"""
        from .strategy_operators import crossover_position_sizing_genes

        return crossover_position_sizing_genes(parent1_ps, parent2_ps)

    @staticmethod
    def _crossover_entry_genes(
        parent1_entry: Optional[EntryGene],
        parent2_entry: Optional[EntryGene],
    ) -> Tuple[Optional[EntryGene], Optional[EntryGene]]:
        """エントリー遺伝子の交叉処理。"""
        from .strategy_operators import crossover_entry_genes

        return crossover_entry_genes(parent1_entry, parent2_entry)

    @classmethod
    def _uniform_crossover(
        cls,
        parent1: "StrategyGene",
        parent2: "StrategyGene",
        config: Any,
    ) -> Tuple["StrategyGene", "StrategyGene"]:
        """ユニフォーム交叉。"""
        from .strategy_operators import uniform_crossover

        return uniform_crossover(cls, parent1, parent2, config)

    @classmethod
    def _single_point_crossover(
        cls,
        parent1: "StrategyGene",
        parent2: "StrategyGene",
        config: Any,
    ) -> Tuple["StrategyGene", "StrategyGene"]:
        """一点交叉。"""
        from .strategy_operators import single_point_crossover

        return single_point_crossover(cls, parent1, parent2, config)
