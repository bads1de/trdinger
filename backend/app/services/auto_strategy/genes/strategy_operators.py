"""
StrategyGene の遺伝的演算ロジック。
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import Any, Optional, Tuple

import numpy as np

from .conditions import ConditionGroup
from .entry import EntryGene
from .position_sizing import (
    PositionSizingGene,
    create_random_position_sizing_gene,
)
from .tpsl import TPSLGene, create_random_tpsl_gene

logger = logging.getLogger(__name__)


def smart_copy(value: Any) -> Any:
    """値をスマートにコピーする。"""
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, list):
        return [smart_copy(item) for item in value]
    if isinstance(value, dict):
        return value.copy()
    return value


def mutate_indicators(mutated, mutation_rate: float, config: Any) -> None:
    """指標遺伝子の突然変異処理。"""
    min_multiplier, max_multiplier = config.indicator_param_mutation_range

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


def mutate_conditions(mutated, mutation_rate: float, config: Any) -> None:
    """条件の突然変異処理。"""

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

    for conditions in (mutated.long_entry_conditions, mutated.short_entry_conditions):
        if random.random() < mutation_threshold:
            maybe_mutate_branch(conditions)


def mutate_strategy_gene(gene, config: Any, mutation_rate: float = 0.1):
    """StrategyGene の突然変異を実行する。"""
    try:
        mutated = gene.clone()

        mutate_indicators(mutated, mutation_rate, config)
        mutate_conditions(mutated, mutation_rate, config)

        min_risk_multiplier, max_risk_multiplier = config.risk_param_mutation_range
        for key, value in mutated.risk_management.items():
            if isinstance(value, (int, float)) and random.random() < mutation_rate:
                if key == "position_size":
                    mutated.risk_management[key] = max(
                        0.01,
                        min(
                            1.0,
                            value
                            * random.uniform(
                                min_risk_multiplier,
                                max_risk_multiplier,
                            ),
                        ),
                    )
                else:
                    mutated.risk_management[key] = value * random.uniform(
                        min_risk_multiplier,
                        max_risk_multiplier,
                    )

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
            sub_gene = getattr(mutated, field_name)
            if sub_gene:
                if random.random() < mutation_rate:
                    setattr(mutated, field_name, sub_gene.mutate(mutation_rate))
            elif random.random() < mutation_rate * creation_prob_mult:
                setattr(mutated, field_name, creator_func())

        if mutated.tool_genes:
            from ..tools import tool_registry

            for tool_gene in mutated.tool_genes:
                if random.random() < mutation_rate:
                    if random.random() < 0.2:
                        tool_gene.enabled = not tool_gene.enabled

                    tool = tool_registry.get(tool_gene.tool_name)
                    if tool:
                        tool_gene.params = tool.mutate_params(tool_gene.params)

        mutated.metadata["mutated"] = True
        mutated.metadata["mutation_rate"] = mutation_rate
        mutated.id = str(uuid.uuid4())

        return mutated

    except Exception as e:
        logger.error(f"戦略遺伝子突然変異エラー: {e}")
        return gene


def adaptive_mutate_strategy_gene(
    gene,
    population,
    config: Any,
    base_mutation_rate: float = 0.1,
):
    """集団分散に基づく適応的突然変異。"""
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

        mutated = mutate_strategy_gene(gene, config, mutation_rate=adaptive_rate)
        mutated.metadata["adaptive_mutation_rate"] = adaptive_rate
        return mutated

    except Exception as e:
        logger.error(f"適応的戦略遺伝子突然変異エラー: {e}")
        return mutate_strategy_gene(
            gene, config, mutation_rate=base_mutation_rate
        )


def crossover_tpsl_genes(
    parent1_tpsl: Optional[TPSLGene],
    parent2_tpsl: Optional[TPSLGene],
) -> Tuple[Optional[TPSLGene], Optional[TPSLGene]]:
    if parent1_tpsl and parent2_tpsl:
        return TPSLGene.crossover(parent1_tpsl, parent2_tpsl)
    if parent1_tpsl:
        return parent1_tpsl, parent1_tpsl.clone()
    if parent2_tpsl:
        return parent2_tpsl, parent2_tpsl.clone()
    return None, None


def crossover_position_sizing_genes(
    parent1_ps: Optional[PositionSizingGene],
    parent2_ps: Optional[PositionSizingGene],
) -> Tuple[Optional[PositionSizingGene], Optional[PositionSizingGene]]:
    if parent1_ps and parent2_ps:
        return PositionSizingGene.crossover(parent1_ps, parent2_ps)
    if parent1_ps:
        return parent1_ps, parent1_ps.clone()
    if parent2_ps:
        return parent2_ps, parent2_ps.clone()
    return None, None


def crossover_entry_genes(
    parent1_entry: Optional[EntryGene],
    parent2_entry: Optional[EntryGene],
) -> Tuple[Optional[EntryGene], Optional[EntryGene]]:
    if parent1_entry and parent2_entry:
        return EntryGene.crossover(parent1_entry, parent2_entry)
    if parent1_entry:
        return parent1_entry, parent1_entry.clone()
    if parent2_entry:
        return parent2_entry, parent2_entry.clone()
    return None, None


def crossover_strategy_genes(
    strategy_gene_class,
    parent1,
    parent2,
    config: Any,
    crossover_type: str = "uniform",
):
    """StrategyGene の交叉処理を実行する。"""
    try:
        if crossover_type == "uniform":
            return uniform_crossover(strategy_gene_class, parent1, parent2, config)
        return single_point_crossover(strategy_gene_class, parent1, parent2, config)
    except Exception as e:
        logger.error(f"戦略遺伝子交叉エラー: {e}")
        return parent1, parent2


def uniform_crossover(strategy_gene_class, parent1, parent2, config: Any):
    """ユニフォーム交叉。"""
    selection_prob = config.crossover_field_selection_probability

    child1_params = {"id": str(uuid.uuid4())}
    child2_params = {"id": str(uuid.uuid4())}

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
            child1_params[field_name] = smart_copy(val1)
            child2_params[field_name] = smart_copy(val2)
        else:
            child1_params[field_name] = smart_copy(val2)
            child2_params[field_name] = smart_copy(val1)

    from .genetic_utils import GeneticUtils

    c1_meta, c2_meta = GeneticUtils.prepare_crossover_metadata(parent1, parent2)
    child1_params["metadata"] = c1_meta
    child2_params["metadata"] = c2_meta

    return strategy_gene_class(**child1_params), strategy_gene_class(**child2_params)


def single_point_crossover(strategy_gene_class, parent1, parent2, config: Any):
    """一点交叉。"""
    min_indicators = min(len(parent1.indicators), len(parent2.indicators))
    crossover_point = 0 if min_indicators <= 1 else random.randint(1, min_indicators)

    c1_ind = [ind.clone() for ind in parent1.indicators[:crossover_point]] + [
        ind.clone() for ind in parent2.indicators[crossover_point:]
    ]
    c2_ind = [ind.clone() for ind in parent2.indicators[:crossover_point]] + [
        ind.clone() for ind in parent1.indicators[crossover_point:]
    ]

    max_indicators = config.max_indicators
    c1_ind = c1_ind[:max_indicators]
    c2_ind = c2_ind[:max_indicators]

    c1_risk = {}
    c2_risk = {}
    all_keys = set(parent1.risk_management.keys()) | set(parent2.risk_management.keys())
    for key in all_keys:
        val1 = parent1.risk_management.get(key, 0)
        val2 = parent2.risk_management.get(key, 0)
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            c1_risk[key] = (val1 + val2) / 2
            c2_risk[key] = (val1 + val2) / 2
        else:
            c1_risk[key] = val1 if random.random() < 0.5 else val2
            c2_risk[key] = val2 if random.random() < 0.5 else val1

    c1_tpsl, c2_tpsl = crossover_tpsl_genes(parent1.tpsl_gene, parent2.tpsl_gene)
    c1_long_tpsl, c2_long_tpsl = crossover_tpsl_genes(
        parent1.long_tpsl_gene,
        parent2.long_tpsl_gene,
    )
    c1_short_tpsl, c2_short_tpsl = crossover_tpsl_genes(
        parent1.short_tpsl_gene,
        parent2.short_tpsl_gene,
    )
    c1_ps, c2_ps = crossover_position_sizing_genes(
        parent1.position_sizing_gene,
        parent2.position_sizing_gene,
    )
    c1_entry, c2_entry = crossover_entry_genes(parent1.entry_gene, parent2.entry_gene)
    c1_long_entry, c2_long_entry = crossover_entry_genes(
        parent1.long_entry_gene,
        parent2.long_entry_gene,
    )
    c1_short_entry, c2_short_entry = crossover_entry_genes(
        parent1.short_entry_gene,
        parent2.short_entry_gene,
    )

    from .genetic_utils import GeneticUtils

    c1_meta, c2_meta = GeneticUtils.prepare_crossover_metadata(parent1, parent2)

    def copy_conditions(conds):
        return [smart_copy(c) for c in conds]

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

    child1 = strategy_gene_class(
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
    child2 = strategy_gene_class(
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
