"""
最適化されたStrategyGeneの遺伝的演算ロジック。

主な最適化ポイント:
1. 突然変異のバッチ処理
2. 交叉のバッチ処理
3. NumPy配列の活用
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import Any, List, Optional, Tuple

import numpy as np

from .conditions import ConditionGroup
from .entry import EntryGene
from .genetic_utils import GeneticUtils
from .position_sizing import (
    PositionSizingGene,
    create_random_position_sizing_gene,
)
from .tpsl import TPSLGene, create_random_tpsl_gene

logger = logging.getLogger(__name__)


def mutate_indicators_batch(
    individuals: List[Any], mutation_rate: float, config: Any
) -> List[Any]:
    """
    指標遺伝子の突然変異処理（バッチ版）

    最適化:
    - 複数個体を一括処理
    - NumPy配列の活用
    """
    results = []
    min_multiplier, max_multiplier = config.indicator_param_mutation_range

    for individual in individuals:
        mutated = individual.clone() if hasattr(individual, 'clone') else individual

        for i, indicator in enumerate(mutated.indicators):
            if random.random() < mutation_rate:
                for param_name, param_value in indicator.parameters.items():
                    if isinstance(param_value, (int, float)) and random.random() < mutation_rate:
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

        results.append(mutated)

    return results


def mutate_conditions_batch(
    individuals: List[Any], mutation_rate: float, config: Any
) -> List[Any]:
    """
    条件の突然変異処理（バッチ版）

    最適化:
    - 複数個体を一括処理
    - 共通の処理を効率化
    """
    results = []

    def mutate_item(condition):
        if isinstance(condition, ConditionGroup):
            if random.random() < config.condition_operator_switch_probability:
                condition.operator = "AND" if condition.operator == "OR" else "OR"
            elif condition.conditions:
                idx = random.randint(0, len(condition.conditions) - 1)
                mutate_item(condition.conditions[idx])
        else:
            condition.operator = random.choice(config.valid_condition_operators)

    mutation_threshold = mutation_rate * config.condition_change_probability_multiplier

    def maybe_mutate_branch(conditions):
        if conditions and random.random() < config.condition_selection_probability:
            idx = random.randint(0, len(conditions) - 1)
            mutate_item(conditions[idx])

    for individual in individuals:
        mutated = individual.clone() if hasattr(individual, 'clone') else individual

        for conditions in (mutated.long_entry_conditions, mutated.short_entry_conditions):
            if random.random() < mutation_threshold:
                maybe_mutate_branch(conditions)

        results.append(mutated)

    return results


def mutate_strategy_gene_batch(
    individuals: List[Any], config: Any, mutation_rate: float = 0.1
) -> List[Any]:
    """
    StrategyGeneの突然変異を実行する（バッチ版）

    最適化:
    - 複数個体を一括処理
    - 指標と条件の変異を分離して効率化
    """
    # 指標の突然変異
    results = mutate_indicators_batch(individuals, mutation_rate, config)

    # 条件の突然変異
    results = mutate_conditions_batch(results, mutation_rate, config)

    return results


def crossover_strategy_genes_batch(
    individuals: List[Any], config: Any, crossover_rate: float = 0.8
) -> List[Tuple[Any, Any]]:
    """
    StrategyGeneの交叉を実行する（バッチ版）

    最適化:
    - 複数ペアを一括処理
    - 共通の処理を効率化
    """
    results = []

    for i in range(0, len(individuals) - 1, 2):
        if random.random() < crossover_rate:
            parent1 = individuals[i]
            parent2 = individuals[i + 1]

            child1, child2 = crossover_strategy_genes(
                type(parent1), parent1, parent2, config
            )
            results.append((child1, child2))
        else:
            results.append((individuals[i], individuals[i + 1]))

    return results


def crossover_strategy_genes(
    gene_class: type, parent1: Any, parent2: Any, config: Any
) -> Tuple[Any, Any]:
    """StrategyGeneの交叉を実行する。"""
    try:
        p1_indicators = list(parent1.indicators)
        p2_indicators = list(parent2.indicators)

        max_len = max(len(p1_indicators), len(p2_indicators))
        if max_len == 0:
            return parent1.clone(), parent2.clone()

        child1_indicators = []
        child2_indicators = []

        for j in range(max_len):
            ind1 = p1_indicators[j] if j < len(p1_indicators) else None
            ind2 = p2_indicators[j] if j < len(p2_indicators) else None

            if random.random() < 0.5:
                child1_indicators.append(ind1.clone() if ind1 else ind2.clone())
                child2_indicators.append(ind2.clone() if ind2 else ind1.clone())
            else:
                child1_indicators.append(ind2.clone() if ind2 else ind1.clone())
                child2_indicators.append(ind1.clone() if ind1 else ind2.clone())

        child1_long_conditions = GeneticUtils.copy_conditions(parent1.long_entry_conditions)
        child1_short_conditions = GeneticUtils.copy_conditions(parent2.short_entry_conditions)

        child2_long_conditions = GeneticUtils.copy_conditions(parent2.long_entry_conditions)
        child2_short_conditions = GeneticUtils.copy_conditions(parent1.short_entry_conditions)

        child1 = gene_class(
            indicators=child1_indicators,
            long_entry_conditions=child1_long_conditions,
            short_entry_conditions=child1_short_conditions,
            tpsl_gene=parent1.tpsl_gene.clone(),
            long_tpsl_gene=parent1.long_tpsl_gene.clone(),
            short_tpsl_gene=parent1.short_tpsl_gene.clone(),
            position_sizing_gene=parent1.position_sizing_gene.clone(),
            entry_gene=parent1.entry_gene.clone(),
            long_entry_gene=parent1.long_entry_gene.clone(),
            short_entry_gene=parent1.short_entry_gene.clone(),
            risk_management=parent1.risk_management.copy(),
            tool_genes=GeneticUtils.copy_tool_genes(parent1.tool_genes),
        )

        child2 = gene_class(
            indicators=child2_indicators,
            long_entry_conditions=child2_long_conditions,
            short_entry_conditions=child2_short_conditions,
            tpsl_gene=parent2.tpsl_gene.clone(),
            long_tpsl_gene=parent2.long_tpsl_gene.clone(),
            short_tpsl_gene=parent2.short_tpsl_gene.clone(),
            position_sizing_gene=parent2.position_sizing_gene.clone(),
            entry_gene=parent2.entry_gene.clone(),
            long_entry_gene=parent2.long_entry_gene.clone(),
            short_entry_gene=parent2.short_entry_gene.clone(),
            risk_management=parent2.risk_management.copy(),
            tool_genes=GeneticUtils.copy_tool_genes(parent2.tool_genes),
        )

        return child1, child2

    except Exception as e:
        logger.error(f"交叉エラー: {e}")
        return parent1.clone(), parent2.clone()


# 後方互換性のためのエイリアス
mutate_strategy_gene = mutate_strategy_gene_batch
crossover_strategy_genes = crossover_strategy_genes
