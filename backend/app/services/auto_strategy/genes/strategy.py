"""
戦略遺伝子モデル
"""

from __future__ import annotations

import copy
import logging
import random
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..config.constants import PositionSizingMethod
from .conditions import Condition, ConditionGroup, StatefulCondition
from .entry import EntryGene
from .indicator import IndicatorGene
from .position_sizing import (
    PositionSizingGene,
    create_random_position_sizing_gene,
    crossover_position_sizing_genes,
    mutate_position_sizing_gene,
)
from .tool import ToolGene
from .tpsl import (
    TPSLGene,
    create_random_tpsl_gene,
    crossover_tpsl_genes,
    mutate_tpsl_gene,
)

logger = logging.getLogger(__name__)


@dataclass
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

    def has_long_short_separation(self) -> bool:
        """ロング・ショート条件が分離されているかチェック（常にTrue）"""
        return True

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

    def mutate(self, config: Any, mutation_rate: float = 0.1) -> StrategyGene:
        """
        戦略遺伝子の突然変異（純粋版）

        自身のコピーを作成して変異させ、新しいインスタンスを返します。
        """
        try:
            # 深いコピーを作成
            mutated = copy.deepcopy(self)

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

            # TP/SL遺伝子の突然変異
            if mutated.tpsl_gene:
                if random.random() < mutation_rate:
                    mutated.tpsl_gene = mutate_tpsl_gene(
                        mutated.tpsl_gene, mutation_rate
                    )
            else:
                if (
                    random.random()
                    < mutation_rate * config.tpsl_gene_creation_probability_multiplier
                ):
                    mutated.tpsl_gene = create_random_tpsl_gene()

            # Long TP/SL遺伝子の突然変異
            if mutated.long_tpsl_gene:
                if random.random() < mutation_rate:
                    mutated.long_tpsl_gene = mutate_tpsl_gene(
                        mutated.long_tpsl_gene, mutation_rate
                    )
            else:
                if (
                    random.random()
                    < mutation_rate * config.tpsl_gene_creation_probability_multiplier
                ):
                    mutated.long_tpsl_gene = create_random_tpsl_gene()

            # Short TP/SL遺伝子の突然変異
            if mutated.short_tpsl_gene:
                if random.random() < mutation_rate:
                    mutated.short_tpsl_gene = mutate_tpsl_gene(
                        mutated.short_tpsl_gene, mutation_rate
                    )
            else:
                if (
                    random.random()
                    < mutation_rate * config.tpsl_gene_creation_probability_multiplier
                ):
                    mutated.short_tpsl_gene = create_random_tpsl_gene()

            # ポジションサイジング遺伝子の突然変異
            if mutated.position_sizing_gene:
                if random.random() < mutation_rate:
                    mutated.position_sizing_gene = mutate_position_sizing_gene(
                        mutated.position_sizing_gene, mutation_rate
                    )
            else:
                if (
                    random.random()
                    < mutation_rate
                    * config.position_sizing_gene_creation_probability_multiplier
                ):
                    mutated.position_sizing_gene = create_random_position_sizing_gene()

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
        戦略遺伝子の交叉
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
    def _mutate_indicators(mutated: StrategyGene, mutation_rate: float, config: Any):
        """指標の突然変異"""
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
                from ..generators.random_gene_generator import RandomGeneGenerator

                generator = RandomGeneGenerator(config)
                new_indicators = (
                    generator.indicator_generator.generate_random_indicators()
                )
                if new_indicators:
                    mutated.indicators.append(random.choice(new_indicators))
            elif len(mutated.indicators) > config.min_indicators and random.random() < (
                1 - config.indicator_add_vs_delete_probability
            ):
                mutated.indicators.pop(random.randint(0, len(mutated.indicators) - 1))

    @staticmethod
    def _mutate_conditions(mutated: StrategyGene, mutation_rate: float, config: Any):
        """条件の突然変異"""

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
            return crossover_tpsl_genes(parent1_tpsl, parent2_tpsl)
        elif parent1_tpsl:
            return parent1_tpsl, copy.deepcopy(parent1_tpsl)
        elif parent2_tpsl:
            return parent2_tpsl, copy.deepcopy(parent2_tpsl)
        else:
            return None, None

    @staticmethod
    def _crossover_position_sizing_genes(
        parent1_ps: Optional[PositionSizingGene],
        parent2_ps: Optional[PositionSizingGene],
    ) -> Tuple[Optional[PositionSizingGene], Optional[PositionSizingGene]]:
        if parent1_ps and parent2_ps:
            return crossover_position_sizing_genes(parent1_ps, parent2_ps)
        elif parent1_ps:
            return parent1_ps, copy.deepcopy(parent1_ps)
        elif parent2_ps:
            return parent2_ps, copy.deepcopy(parent2_ps)
        else:
            return None, None

    @classmethod
    def _uniform_crossover(
        cls, parent1: StrategyGene, parent2: StrategyGene, config: Any
    ) -> Tuple[StrategyGene, StrategyGene]:
        """ユニフォーム交叉"""
        selection_prob = config.crossover_field_selection_probability

        # 各フィールドを確率的に選択
        c1_ind = (
            parent1.indicators
            if random.random() < selection_prob
            else parent2.indicators
        )
        c2_ind = (
            parent2.indicators
            if random.random() < selection_prob
            else parent1.indicators
        )

        c1_long = (
            parent1.long_entry_conditions
            if random.random() < selection_prob
            else parent2.long_entry_conditions
        )
        c2_long = (
            parent2.long_entry_conditions
            if random.random() < selection_prob
            else parent1.long_entry_conditions
        )

        c1_short = (
            parent1.short_entry_conditions
            if random.random() < selection_prob
            else parent2.short_entry_conditions
        )
        c2_short = (
            parent2.short_entry_conditions
            if random.random() < selection_prob
            else parent1.short_entry_conditions
        )

        c1_risk = (
            parent1.risk_management
            if random.random() < selection_prob
            else parent2.risk_management
        )
        c2_risk = (
            parent2.risk_management
            if random.random() < selection_prob
            else parent1.risk_management
        )

        c1_tpsl = (
            parent1.tpsl_gene if random.random() < selection_prob else parent2.tpsl_gene
        )
        c2_tpsl = (
            parent2.tpsl_gene if random.random() < selection_prob else parent1.tpsl_gene
        )

        c1_long_tpsl = (
            parent1.long_tpsl_gene
            if random.random() < selection_prob
            else parent2.long_tpsl_gene
        )
        c2_long_tpsl = (
            parent2.long_tpsl_gene
            if random.random() < selection_prob
            else parent1.long_tpsl_gene
        )

        c1_short_tpsl = (
            parent1.short_tpsl_gene
            if random.random() < selection_prob
            else parent2.short_tpsl_gene
        )
        c2_short_tpsl = (
            parent2.short_tpsl_gene
            if random.random() < selection_prob
            else parent1.short_tpsl_gene
        )

        c1_ps = (
            parent1.position_sizing_gene
            if random.random() < selection_prob
            else parent2.position_sizing_gene
        )
        c2_ps = (
            parent2.position_sizing_gene
            if random.random() < selection_prob
            else parent1.position_sizing_gene
        )

        c1_tool = (
            copy.deepcopy(parent1.tool_genes)
            if random.random() < selection_prob
            else copy.deepcopy(parent2.tool_genes)
        )
        c2_tool = (
            copy.deepcopy(parent2.tool_genes)
            if random.random() < selection_prob
            else copy.deepcopy(parent1.tool_genes)
        )

        from .genetic_utils import GeneticUtils

        c1_meta, c2_meta = GeneticUtils.prepare_crossover_metadata(parent1, parent2)

        child1 = cls(
            id=str(uuid.uuid4()),
            indicators=c1_ind,
            long_entry_conditions=c1_long,
            short_entry_conditions=c1_short,
            risk_management=c1_risk,
            tpsl_gene=c1_tpsl,
            long_tpsl_gene=c1_long_tpsl,
            short_tpsl_gene=c1_short_tpsl,
            position_sizing_gene=c1_ps,
            tool_genes=c1_tool,
            metadata=c1_meta,
        )

        child2 = cls(
            id=str(uuid.uuid4()),
            indicators=c2_ind,
            long_entry_conditions=c2_long,
            short_entry_conditions=c2_short,
            risk_management=c2_risk,
            tpsl_gene=c2_tpsl,
            long_tpsl_gene=c2_long_tpsl,
            short_tpsl_gene=c2_short_tpsl,
            position_sizing_gene=c2_ps,
            tool_genes=c2_tool,
            metadata=c2_meta,
        )

        return child1, child2

    @classmethod
    def _single_point_crossover(
        cls, parent1: StrategyGene, parent2: StrategyGene, config: Any
    ) -> Tuple[StrategyGene, StrategyGene]:
        """一点交叉"""
        min_indicators = min(len(parent1.indicators), len(parent2.indicators))
        crossover_point = 0 if min_indicators <= 1 else random.randint(1, min_indicators)

        c1_ind = (
            parent1.indicators[:crossover_point] + parent2.indicators[crossover_point:]
        )
        c2_ind = (
            parent2.indicators[:crossover_point] + parent1.indicators[crossover_point:]
        )

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

        from .genetic_utils import GeneticUtils

        c1_meta, c2_meta = GeneticUtils.prepare_crossover_metadata(parent1, parent2)

        if random.random() < 0.5:
            c1_long_cond = parent1.long_entry_conditions.copy()
            c2_long_cond = parent2.long_entry_conditions.copy()
        else:
            c1_long_cond = parent2.long_entry_conditions.copy()
            c2_long_cond = parent1.long_entry_conditions.copy()

        if random.random() < 0.5:
            c1_short_cond = parent1.short_entry_conditions.copy()
            c2_short_cond = parent2.short_entry_conditions.copy()
        else:
            c1_short_cond = parent2.short_entry_conditions.copy()
            c2_short_cond = parent1.short_entry_conditions.copy()

        c1_tool = (
            copy.deepcopy(parent1.tool_genes)
            if random.random() < 0.5
            else copy.deepcopy(parent2.tool_genes)
        )
        c2_tool = (
            copy.deepcopy(parent2.tool_genes)
            if random.random() < 0.5
            else copy.deepcopy(parent1.tool_genes)
        )

        child1 = cls(
            id=str(uuid.uuid4()),
            indicators=c1_ind,
            long_entry_conditions=c1_long_cond,
            short_entry_conditions=c1_short_cond,
            risk_management=c1_risk,
            tpsl_gene=c1_tpsl,
            long_tpsl_gene=c1_long_tpsl,
            short_tpsl_gene=c1_short_tpsl,
            position_sizing_gene=c1_ps,
            tool_genes=c1_tool,
            metadata=c1_meta,
        )
        child2 = cls(
            id=str(uuid.uuid4()),
            indicators=c2_ind,
            long_entry_conditions=c2_long_cond,
            short_entry_conditions=c2_short_cond,
            risk_management=c2_risk,
            tpsl_gene=c2_tpsl,
            long_tpsl_gene=c2_long_tpsl,
            short_tpsl_gene=c2_short_tpsl,
            position_sizing_gene=c2_ps,
            tool_genes=c2_tool,
            metadata=c2_meta,
        )

        return child1, child2

