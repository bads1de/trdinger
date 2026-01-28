"""
戦略遺伝子モデル
"""

from __future__ import annotations


import logging
import random
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


from .conditions import Condition, ConditionGroup, StatefulCondition
from .entry import EntryGene
from .indicator import IndicatorGene
from .position_sizing import (
    PositionSizingGene,
    create_random_position_sizing_gene,
)
from .tool import ToolGene
from .tpsl import (
    TPSLGene,
    create_random_tpsl_gene,
)

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
    # ツール遺伝子（エントリーフィルターなど）
    tool_genes: List[ToolGene] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_default(cls) -> "StrategyGene":
        """
        デフォルトの戦略遺伝子を作成
        """
        from .tpsl import TPSLGene
        from .position_sizing import PositionSizingGene, PositionSizingMethod
        from .indicator import IndicatorGene
        from .conditions import Condition

        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ]

        return cls(
            id=str(uuid.uuid4()),
            indicators=indicators,
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="open")
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="open")
            ],
            risk_management={"position_size": 0.1},
            tpsl_gene=TPSLGene(take_profit_pct=0.01, stop_loss_pct=0.005, enabled=True),
            position_sizing_gene=PositionSizingGene(
                method=PositionSizingMethod.FIXED_QUANTITY,
                fixed_quantity=1000,
                enabled=True,
            ),
            metadata={"generated_by": "create_default"},
        )

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
        """
        パーツから戦略遺伝子を組み立てる

        提供された各コンポーネントを組み合わせて、新しいStrategyGeneインスタンスを生成します。
        IDは新しく採番され、メタデータにアセンブル時刻が追加されます。

        Returns:
            構築されたStrategyGene
        """
        final_metadata = metadata or {}
        final_metadata.setdefault("assembled_at", datetime.now().isoformat())

        return cls(
            id=str(uuid.uuid4()),
            indicators=indicators,
            long_entry_conditions=long_entry_conditions,
            short_entry_conditions=short_entry_conditions,
            tpsl_gene=tpsl_gene,
            long_tpsl_gene=long_tpsl_gene,
            short_tpsl_gene=short_tpsl_gene,
            position_sizing_gene=position_sizing_gene,
            entry_gene=entry_gene,
            long_entry_gene=long_entry_gene,
            short_entry_gene=short_entry_gene,
            tool_genes=tool_genes or [],
            risk_management=risk_management or {"position_size": 0.1},
            metadata=final_metadata,
        )

    def has_long_short_separation(self) -> bool:
        """ロング・ショート条件が分離されているかチェック（常にTrue）"""
        return True

    def validate(self) -> Tuple[bool, List[str]]:
        """戦略遺伝子の妥当性を検証し、(is_valid, errors) を返す"""
        from .validator import GeneValidator

        validator = GeneValidator()
        is_valid, errors = validator.validate_strategy_gene(self)
        return is_valid, errors

    @staticmethod
    def _smart_copy(value: Any) -> Any:
        """値をスマートにコピー（cloneメソッドがあれば使用）"""
        if hasattr(value, "clone"):
            return value.clone()
        if isinstance(value, list):
            return [StrategyGene._smart_copy(item) for item in value]
        if isinstance(value, dict):
            return value.copy()
        return value

    def clone(self, keep_id: bool = False) -> StrategyGene:
        """軽量コピーを作成"""
        return StrategyGene(
            id=self.id if keep_id else str(uuid.uuid4()),
            indicators=[ind.clone() for ind in self.indicators],
            long_entry_conditions=[
                self._smart_copy(c) for c in self.long_entry_conditions
            ],
            short_entry_conditions=[
                self._smart_copy(c) for c in self.short_entry_conditions
            ],
            stateful_conditions=[c.clone() for c in self.stateful_conditions],
            risk_management=self.risk_management.copy(),
            tpsl_gene=self.tpsl_gene.clone() if self.tpsl_gene else None,
            long_tpsl_gene=(
                self.long_tpsl_gene.clone() if self.long_tpsl_gene else None
            ),
            short_tpsl_gene=(
                self.short_tpsl_gene.clone() if self.short_tpsl_gene else None
            ),
            position_sizing_gene=(
                self.position_sizing_gene.clone() if self.position_sizing_gene else None
            ),
            entry_gene=self.entry_gene.clone() if self.entry_gene else None,
            long_entry_gene=(
                self.long_entry_gene.clone() if self.long_entry_gene else None
            ),
            short_entry_gene=(
                self.short_entry_gene.clone() if self.short_entry_gene else None
            ),
            tool_genes=[t.clone() for t in self.tool_genes],
            metadata=self.metadata.copy(),
        )

    def mutate(self, config: Any, mutation_rate: float = 0.1) -> StrategyGene:
        """
        戦略遺伝子の突然変異を実行します。

        自身のコピーを作成し、各サブコンポーネント（指標、条件、リスク管理、TPSLなど）
        に対して確率的に変異を適用します。

        Args:
            config: GA設定（パラメータ範囲や変異確率の倍率を含む）
            mutation_rate: 基本となる突然変異率 (0.0 - 1.0)

        Returns:
            突然変異が適用された新しいStrategyGeneインスタンス
        """
        try:
            # 深いコピーを作成
            mutated = self.clone()

            # 指標遺伝子の突然変異
            self._mutate_indicators(mutated, mutation_rate, config)

            # 条件の突然変異
            self._mutate_conditions(mutated, mutation_rate, config)

            # リスク管理設定の突然変異
            min_risk_multiplier, max_risk_multiplier = config.risk_param_mutation_range
            for key, value in mutated.risk_management.items():
                if isinstance(value, (int, float)) and random.random() < mutation_rate:
                    if key == "position_size":
                        # ポジションサイズの場合
                        mutated.risk_management[key] = max(
                            0.01,
                            min(
                                1.0,
                                value
                                * random.uniform(
                                    min_risk_multiplier, max_risk_multiplier
                                ),
                            ),
                        )
                    else:
                        # その他の数値設定
                        mutated.risk_management[key] = value * random.uniform(
                            min_risk_multiplier, max_risk_multiplier
                        )

            # サブ遺伝子の突然変異処理
            gene_fields = [
                (
                    "tpsl_gene",
                    create_random_tpsl_gene,
                    config.tpsl_gene_creation_probability_multiplier,
                ),
                (
                    "long_tpsl_gene",
                    create_random_tpsl_gene,
                    config.tpsl_gene_creation_probability_multiplier,
                ),
                (
                    "short_tpsl_gene",
                    create_random_tpsl_gene,
                    config.tpsl_gene_creation_probability_multiplier,
                ),
                (
                    "position_sizing_gene",
                    create_random_position_sizing_gene,
                    config.position_sizing_gene_creation_probability_multiplier,
                ),
            ]

            for field_name, creator_func, creation_prob_mult in gene_fields:
                gene = getattr(mutated, field_name)
                if gene:
                    if random.random() < mutation_rate:
                        # BaseGeneを継承しているため .mutate() が利用可能
                        setattr(mutated, field_name, gene.mutate(mutation_rate))
                else:
                    if random.random() < mutation_rate * creation_prob_mult:
                        setattr(mutated, field_name, creator_func())

            # ツール遺伝子の突然変異
            if mutated.tool_genes:
                from ..tools import tool_registry

                for tool_gene in mutated.tool_genes:
                    if random.random() < mutation_rate:
                        # 有効/無効を反転（20%の確率）
                        if random.random() < 0.2:
                            tool_gene.enabled = not tool_gene.enabled

                        # ツール固有のパラメータ変異
                        tool = tool_registry.get(tool_gene.tool_name)
                        if tool:
                            tool_gene.params = tool.mutate_params(tool_gene.params)

            # メタデータの更新
            mutated.metadata["mutated"] = True
            mutated.metadata["mutation_rate"] = mutation_rate
            mutated.id = str(uuid.uuid4())

            return mutated

        except Exception as e:
            logger.error(f"戦略遺伝子突然変異エラー: {e}")
            return self

    def adaptive_mutate(
        self, population: List[Any], config: Any, base_mutation_rate: float = 0.1
    ) -> StrategyGene:
        """
        適応的突然変異

        集団の多様性に基づいて変異率を調整します。
        """
        try:
            fitnesses = []
            for ind in population:
                if hasattr(ind, "fitness") and ind.fitness and ind.fitness.values:
                    fitnesses.append(ind.fitness.values[0])

            if not fitnesses:
                adaptive_rate = base_mutation_rate
            else:
                variance = np.var(fitnesses)
                variance_threshold = config.adaptive_mutation_variance_threshold

                if variance > variance_threshold:
                    adaptive_rate = (
                        base_mutation_rate
                        * config.adaptive_mutation_rate_decrease_multiplier
                    )
                else:
                    adaptive_rate = (
                        base_mutation_rate
                        * config.adaptive_mutation_rate_increase_multiplier
                    )

                adaptive_rate = max(0.01, min(1.0, adaptive_rate))

            mutated = self.mutate(config, mutation_rate=adaptive_rate)
            mutated.metadata["adaptive_mutation_rate"] = adaptive_rate
            return mutated

        except Exception as e:
            logger.error(f"適応的戦略遺伝子突然変異エラー: {e}")
            return self.mutate(config, mutation_rate=base_mutation_rate)

    @classmethod
    def crossover(
        cls,
        parent1: StrategyGene,
        parent2: StrategyGene,
        config: Any,
        crossover_type: str = "uniform",
    ) -> Tuple[StrategyGene, StrategyGene]:
        """
        2つの親個体から交叉により新しい子個体を生成します。

        Args:
            parent1: 親個体1
            parent2: 親個体2
            config: GA設定
            crossover_type: 交叉の種類 ("uniform" または "single_point")

        Returns:
            生成された2つの子個体のタプル
        """
        try:
            if crossover_type == "uniform":
                return cls._uniform_crossover(parent1, parent2, config)
            else:
                # 一点交叉（既存ロジック）
                return cls._single_point_crossover(parent1, parent2, config)

        except Exception as e:
            logger.error(f"戦略遺伝子交叉エラー: {e}")
            return parent1, parent2

    @staticmethod
    def _mutate_indicators(mutated: "StrategyGene", mutation_rate: float, config: Any):
        """
        指標遺伝子の突然変異処理

        1. 確率的に既存指標のパラメータ（period等）を変更します。
        2. 確率的に指標の新規追加、または既存指標の削除を行います。
        """
        min_multiplier, max_multiplier = config.indicator_param_mutation_range
        # 指標パラメータの突然変異
        for i, indicator in enumerate(mutated.indicators):
            if random.random() < mutation_rate:
                for param_name, param_value in indicator.parameters.items():
                    if (
                        isinstance(param_value, (int, float))
                        and random.random() < mutation_rate
                    ):
                        if (
                            param_name == "period"
                            and hasattr(config, "parameter_ranges")
                            and "period" in config.parameter_ranges
                        ):
                            min_p, max_p = config.parameter_ranges["period"]
                            mutated.indicators[i].parameters[param_name] = max(
                                min_p,
                                min(
                                    max_p,
                                    int(
                                        param_value
                                        * random.uniform(min_multiplier, max_multiplier)
                                    ),
                                ),
                            )
                        else:
                            mutated.indicators[i].parameters[param_name] = (
                                param_value
                                * random.uniform(min_multiplier, max_multiplier)
                            )

        # 指標の追加・削除
        if random.random() < mutation_rate * config.indicator_add_delete_probability:
            max_indicators = config.max_indicators
            if (
                len(mutated.indicators) < max_indicators
                and random.random() < config.indicator_add_vs_delete_probability
            ):
                from .indicator import generate_random_indicators

                new_indicators = generate_random_indicators(config)
                if new_indicators:
                    mutated.indicators.append(random.choice(new_indicators))
            elif len(mutated.indicators) > config.min_indicators and random.random() < (
                1 - config.indicator_add_vs_delete_probability
            ):
                mutated.indicators.pop(random.randint(0, len(mutated.indicators) - 1))

    @staticmethod
    def _mutate_conditions(mutated: "StrategyGene", mutation_rate: float, config: Any):
        """
        取引条件の突然変異処理

        ロング/ショートそれぞれの条件リストに対して、
        演算子の切り替えや条件の書き換えを確率的に実行します。
        """

        def mutate_item(condition):
            if isinstance(condition, ConditionGroup):
                if random.random() < config.condition_operator_switch_probability:
                    condition.operator = "AND" if condition.operator == "OR" else "OR"
                else:
                    if condition.conditions:
                        idx = random.randint(0, len(condition.conditions) - 1)
                        mutate_item(condition.conditions[idx])
            else:
                condition.operator = random.choice(config.valid_condition_operators)

        if (
            random.random()
            < mutation_rate * config.condition_change_probability_multiplier
        ):
            if (
                mutated.long_entry_conditions
                and random.random() < config.condition_selection_probability
            ):
                idx = random.randint(0, len(mutated.long_entry_conditions) - 1)
                mutate_item(mutated.long_entry_conditions[idx])

        if (
            random.random()
            < mutation_rate * config.condition_change_probability_multiplier
        ):
            if (
                mutated.short_entry_conditions
                and random.random() < config.condition_selection_probability
            ):
                idx = random.randint(0, len(mutated.short_entry_conditions) - 1)
                mutate_item(mutated.short_entry_conditions[idx])

    @staticmethod
    def _crossover_tpsl_genes(
        parent1_tpsl: Optional[TPSLGene], parent2_tpsl: Optional[TPSLGene]
    ) -> Tuple[Optional[TPSLGene], Optional[TPSLGene]]:
        if parent1_tpsl and parent2_tpsl:
            return TPSLGene.crossover(parent1_tpsl, parent2_tpsl)
        elif parent1_tpsl:
            return parent1_tpsl, parent1_tpsl.clone()
        elif parent2_tpsl:
            return parent2_tpsl, parent2_tpsl.clone()
        else:
            return None, None

    @staticmethod
    def _crossover_position_sizing_genes(
        parent1_ps: Optional[PositionSizingGene],
        parent2_ps: Optional[PositionSizingGene],
    ) -> Tuple[Optional[PositionSizingGene], Optional[PositionSizingGene]]:
        if parent1_ps and parent2_ps:
            return PositionSizingGene.crossover(parent1_ps, parent2_ps)
        elif parent1_ps:
            return parent1_ps, parent1_ps.clone()
        elif parent2_ps:
            return parent2_ps, parent2_ps.clone()
        else:
            return None, None

    @staticmethod
    def _crossover_entry_genes(
        parent1_entry: Optional[EntryGene], parent2_entry: Optional[EntryGene]
    ) -> Tuple[Optional[EntryGene], Optional[EntryGene]]:
        if parent1_entry and parent2_entry:
            return EntryGene.crossover(parent1_entry, parent2_entry)
        elif parent1_entry:
            return parent1_entry, parent1_entry.clone()
        elif parent2_entry:
            return parent2_entry, parent2_entry.clone()
        else:
            return None, None

    @classmethod
    def _uniform_crossover(
        cls, parent1: StrategyGene, parent2: StrategyGene, config: Any
    ) -> Tuple[StrategyGene, StrategyGene]:
        """ユニフォーム交叉"""
        selection_prob = config.crossover_field_selection_probability

        child1_params = {"id": str(uuid.uuid4())}
        child2_params = {"id": str(uuid.uuid4())}

        # 交叉対象のフィールド
        fields = [
            "indicators",
            "long_entry_conditions",
            "short_entry_conditions",
            "stateful_conditions",
            "risk_management",
            "tpsl_gene",
            "long_tpsl_gene",
            "short_tpsl_gene",
            "position_sizing_gene",
            "entry_gene",
            "long_entry_gene",
            "short_entry_gene",
            "tool_genes",
        ]

        for field_name in fields:
            val1 = getattr(parent1, field_name)
            val2 = getattr(parent2, field_name)

            if random.random() < selection_prob:
                child1_params[field_name] = cls._smart_copy(val1)
                child2_params[field_name] = cls._smart_copy(val2)
            else:
                child1_params[field_name] = cls._smart_copy(val2)
                child2_params[field_name] = cls._smart_copy(val1)

        from .genetic_utils import GeneticUtils

        c1_meta, c2_meta = GeneticUtils.prepare_crossover_metadata(parent1, parent2)

        child1_params["metadata"] = c1_meta
        child2_params["metadata"] = c2_meta

        return cls(**child1_params), cls(**child2_params)

    @classmethod
    def _single_point_crossover(
        cls, parent1: StrategyGene, parent2: StrategyGene, config: Any
    ) -> Tuple[StrategyGene, StrategyGene]:
        """一点交叉"""
        min_indicators = min(len(parent1.indicators), len(parent2.indicators))
        crossover_point = (
            0 if min_indicators <= 1 else random.randint(1, min_indicators)
        )

        c1_ind = [ind.clone() for ind in parent1.indicators[:crossover_point]] + [
            ind.clone() for ind in parent2.indicators[crossover_point:]
        ]
        c2_ind = [ind.clone() for ind in parent2.indicators[:crossover_point]] + [
            ind.clone() for ind in parent1.indicators[crossover_point:]
        ]

        max_indicators = config.max_indicators
        c1_ind = c1_ind[:max_indicators]
        c2_ind = c2_ind[:max_indicators]

        # リスク管理の平均化
        c1_risk = {}
        c2_risk = {}
        all_keys = set(parent1.risk_management.keys()) | set(
            parent2.risk_management.keys()
        )
        for key in all_keys:
            val1 = parent1.risk_management.get(key, 0)
            val2 = parent2.risk_management.get(key, 0)
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                c1_risk[key] = (val1 + val2) / 2
                c2_risk[key] = (val1 + val2) / 2
            else:
                c1_risk[key] = val1 if random.random() < 0.5 else val2
                c2_risk[key] = val2 if random.random() < 0.5 else val1

        c1_tpsl, c2_tpsl = cls._crossover_tpsl_genes(
            parent1.tpsl_gene, parent2.tpsl_gene
        )
        c1_long_tpsl, c2_long_tpsl = cls._crossover_tpsl_genes(
            parent1.long_tpsl_gene, parent2.long_tpsl_gene
        )
        c1_short_tpsl, c2_short_tpsl = cls._crossover_tpsl_genes(
            parent1.short_tpsl_gene, parent2.short_tpsl_gene
        )
        c1_ps, c2_ps = cls._crossover_position_sizing_genes(
            parent1.position_sizing_gene, parent2.position_sizing_gene
        )
        c1_entry, c2_entry = cls._crossover_entry_genes(
            parent1.entry_gene, parent2.entry_gene
        )
        c1_long_entry, c2_long_entry = cls._crossover_entry_genes(
            parent1.long_entry_gene, parent2.long_entry_gene
        )
        c1_short_entry, c2_short_entry = cls._crossover_entry_genes(
            parent1.short_entry_gene, parent2.short_entry_gene
        )

        from .genetic_utils import GeneticUtils

        c1_meta, c2_meta = GeneticUtils.prepare_crossover_metadata(parent1, parent2)

        def copy_conditions(conds):
            return [cls._smart_copy(c) for c in conds]

        if random.random() < 0.5:
            c1_long_cond = copy_conditions(parent1.long_entry_conditions)
            c2_long_cond = copy_conditions(parent2.long_entry_conditions)
        else:
            c1_long_cond = copy_conditions(parent2.long_entry_conditions)
            c2_long_cond = copy_conditions(parent1.long_entry_conditions)

        if random.random() < 0.5:
            c1_short_cond = copy_conditions(parent1.short_entry_conditions)
            c2_short_cond = copy_conditions(parent2.short_entry_conditions)
        else:
            c1_short_cond = copy_conditions(parent2.short_entry_conditions)
            c2_short_cond = copy_conditions(parent1.short_entry_conditions)

        def copy_stateful(conds):
            return [c.clone() for c in conds]

        if random.random() < 0.5:
            c1_stateful = copy_stateful(parent1.stateful_conditions)
            c2_stateful = copy_stateful(parent2.stateful_conditions)
        else:
            c1_stateful = copy_stateful(parent2.stateful_conditions)
            c2_stateful = copy_stateful(parent1.stateful_conditions)

        def copy_tools(tools):
            return [t.clone() for t in tools]

        c1_tool = (
            copy_tools(parent1.tool_genes)
            if random.random() < 0.5
            else copy_tools(parent2.tool_genes)
        )
        c2_tool = (
            copy_tools(parent2.tool_genes)
            if random.random() < 0.5
            else copy_tools(parent1.tool_genes)
        )

        child1 = cls(
            id=str(uuid.uuid4()),
            indicators=c1_ind,
            long_entry_conditions=c1_long_cond,
            short_entry_conditions=c1_short_cond,
            stateful_conditions=c1_stateful,
            risk_management=c1_risk,
            tpsl_gene=c1_tpsl,
            long_tpsl_gene=c1_long_tpsl,
            short_tpsl_gene=c1_short_tpsl,
            position_sizing_gene=c1_ps,
            entry_gene=c1_entry,
            long_entry_gene=c1_long_entry,
            short_entry_gene=c1_short_entry,
            tool_genes=c1_tool,
            metadata=c1_meta,
        )
        child2 = cls(
            id=str(uuid.uuid4()),
            indicators=c2_ind,
            long_entry_conditions=c2_long_cond,
            short_entry_conditions=c2_short_cond,
            stateful_conditions=c2_stateful,
            risk_management=c2_risk,
            tpsl_gene=c2_tpsl,
            long_tpsl_gene=c2_long_tpsl,
            short_tpsl_gene=c2_short_tpsl,
            position_sizing_gene=c2_ps,
            entry_gene=c2_entry,
            long_entry_gene=c2_long_entry,
            short_entry_gene=c2_short_entry,
            tool_genes=c2_tool,
            metadata=c2_meta,
        )

        return child1, child2
