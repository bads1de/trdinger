"""
StrategyGene の交叉（crossover）演算ロジック。

遺伝的アルゴリズムにおけるユニフォーム交叉・一点交叉を提供します。
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import Any, List, Optional, Tuple

from ..entry import EntryGene
from ..exit import ExitGene
from ..genetic_utils import GeneticUtils
from ..position_sizing import PositionSizingGene
from ..tpsl import TPSLGene

logger = logging.getLogger(__name__)


def crossover_tpsl_genes(
    parent1_tpsl: Optional[TPSLGene],
    parent2_tpsl: Optional[TPSLGene],
) -> Tuple[Optional[TPSLGene], Optional[TPSLGene]]:
    """TPSL遺伝子の交叉を実行する。"""
    return GeneticUtils.crossover_optional_gene(parent1_tpsl, parent2_tpsl, TPSLGene)


def crossover_position_sizing_genes(
    parent1_ps: Optional[PositionSizingGene],
    parent2_ps: Optional[PositionSizingGene],
) -> Tuple[Optional[PositionSizingGene], Optional[PositionSizingGene]]:
    """ポジションサイジング遺伝子の交叉を実行する。"""
    return GeneticUtils.crossover_optional_gene(
        parent1_ps, parent2_ps, PositionSizingGene
    )


def crossover_entry_genes(
    parent1_entry: Optional[EntryGene],
    parent2_entry: Optional[EntryGene],
) -> Tuple[Optional[EntryGene], Optional[EntryGene]]:
    """エントリー遺伝子の交叉を実行する。"""
    return GeneticUtils.crossover_optional_gene(parent1_entry, parent2_entry, EntryGene)


def crossover_exit_genes(
    parent1_exit: Optional[ExitGene],
    parent2_exit: Optional[ExitGene],
) -> Tuple[Optional[ExitGene], Optional[ExitGene]]:
    """イグジット遺伝子の交叉を実行する。"""
    return GeneticUtils.crossover_optional_gene(parent1_exit, parent2_exit, ExitGene)


def crossover_strategy_genes(
    strategy_gene_class,
    parent1,
    parent2,
    config: Any,
    crossover_type: str = "uniform",
):
    """2つの親個体から新しい2つの子個体を交叉により生成する。"""
    try:
        if crossover_type == "uniform":
            return uniform_crossover(strategy_gene_class, parent1, parent2, config)
        return single_point_crossover(strategy_gene_class, parent1, parent2, config)
    except Exception as e:
        logger.error(f"戦略遺伝子交叉エラー: {e}")
        child1 = parent1.clone() if hasattr(parent1, "clone") else parent1
        child2 = parent2.clone() if hasattr(parent2, "clone") else parent2
        for child in (child1, child2):
            if hasattr(child, "fitness"):
                del child.fitness.values
        return child1, child2


def crossover_strategy_genes_batch(
    individuals: List[Any], config: Any, crossover_rate: float = 0.8
) -> List[Tuple[Any, Any]]:
    """StrategyGene の交叉をバッチで実行する。"""
    results: List[Tuple[Any, Any]] = []
    num_individuals = len(individuals)
    last_pair_index = (
        num_individuals - 1 if num_individuals % 2 == 0 else num_individuals - 2
    )

    for i in range(0, last_pair_index, 2):
        if random.random() < crossover_rate:
            parent1 = individuals[i]
            parent2 = individuals[i + 1]
            child1, child2 = crossover_strategy_genes(
                type(parent1),
                parent1,
                parent2,
                config,
            )
            results.append((child1, child2))
        else:
            results.append((individuals[i], individuals[i + 1]))

    if num_individuals % 2 == 1:
        last_individual = individuals[-1]
        results.append((last_individual, last_individual))

    return results


def uniform_crossover(strategy_gene_class, parent1, parent2, config: Any):
    """ユニフォーム交叉（一様交叉）。"""
    selection_prob = config.mutation_config.crossover_field_selection_probability

    child1_params: dict[str, Any] = {"id": str(uuid.uuid4())}
    child2_params: dict[str, Any] = {"id": str(uuid.uuid4())}

    fields = (
        strategy_gene_class.crossover_field_names()
        if hasattr(strategy_gene_class, "crossover_field_names")
        else ()
    )

    for field_name in fields:
        val1 = getattr(parent1, field_name)
        val2 = getattr(parent2, field_name)

        if random.random() < selection_prob:
            child1_params[field_name] = GeneticUtils.smart_copy(val1)
            child2_params[field_name] = GeneticUtils.smart_copy(val2)
        else:
            child1_params[field_name] = GeneticUtils.smart_copy(val2)
            child2_params[field_name] = GeneticUtils.smart_copy(val1)

    c1_meta, c2_meta = GeneticUtils.prepare_crossover_metadata(parent1, parent2)
    child1_params["metadata"] = c1_meta
    child2_params["metadata"] = c2_meta

    return strategy_gene_class(**child1_params), strategy_gene_class(**child2_params)


def single_point_crossover(strategy_gene_class, parent1, parent2, config: Any):
    """一点交叉。"""
    max_indicators_parent1 = len(parent1.indicators)
    max_indicators_parent2 = len(parent2.indicators)
    max_indicators = config.max_indicators
    min_indicators = min(max_indicators_parent1, max_indicators_parent2)

    if min_indicators <= 0:
        crossover_point = 0
    elif min_indicators == 1:
        crossover_point = random.randint(0, 1)
    else:
        crossover_point = random.randint(1, min_indicators)

    c1_ind = [ind.clone() for ind in parent1.indicators[:crossover_point]] + [
        ind.clone() for ind in parent2.indicators[crossover_point:]
    ]
    c2_ind = [ind.clone() for ind in parent2.indicators[:crossover_point]] + [
        ind.clone() for ind in parent1.indicators[crossover_point:]
    ]

    c1_ind = c1_ind[:max_indicators]
    c2_ind = c2_ind[:max_indicators]

    c1_risk = {}
    c2_risk = {}
    all_keys = set(parent1.risk_management.keys()) | set(parent2.risk_management.keys())
    for key in all_keys:
        val1 = parent1.risk_management.get(key, 0)
        val2 = parent2.risk_management.get(key, 0)
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if random.random() < 0.5:
                c1_risk[key] = val1
                c2_risk[key] = val2
            else:
                c1_risk[key] = val2
                c2_risk[key] = val1
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
    c1_exit, c2_exit = crossover_exit_genes(parent1.exit_gene, parent2.exit_gene)

    c1_meta, c2_meta = GeneticUtils.prepare_crossover_metadata(parent1, parent2)

    if random.random() < 0.5:
        c1_long_cond = GeneticUtils.copy_conditions(parent1.long_entry_conditions)
        c2_long_cond = GeneticUtils.copy_conditions(parent2.long_entry_conditions)
    else:
        c1_long_cond = GeneticUtils.copy_conditions(parent2.long_entry_conditions)
        c2_long_cond = GeneticUtils.copy_conditions(parent1.long_entry_conditions)

    if random.random() < 0.5:
        c1_short_cond = GeneticUtils.copy_conditions(parent1.short_entry_conditions)
        c2_short_cond = GeneticUtils.copy_conditions(parent2.short_entry_conditions)
    else:
        c1_short_cond = GeneticUtils.copy_conditions(parent2.short_entry_conditions)
        c2_short_cond = GeneticUtils.copy_conditions(parent1.short_entry_conditions)

    if random.random() < 0.5:
        c1_stateful = GeneticUtils.copy_stateful_conditions(parent1.stateful_conditions)
        c2_stateful = GeneticUtils.copy_stateful_conditions(parent2.stateful_conditions)
    else:
        c1_stateful = GeneticUtils.copy_stateful_conditions(parent2.stateful_conditions)
        c2_stateful = GeneticUtils.copy_stateful_conditions(parent1.stateful_conditions)

    if random.random() < 0.5:
        c1_tool = GeneticUtils.copy_tool_genes(parent1.tool_genes)
        c2_tool = GeneticUtils.copy_tool_genes(parent2.tool_genes)
    else:
        c1_tool = GeneticUtils.copy_tool_genes(parent2.tool_genes)
        c2_tool = GeneticUtils.copy_tool_genes(parent1.tool_genes)

    # フィルター数制限を強制
    from ...generators.random_gene_generator import RandomGeneGenerator
    generator = RandomGeneGenerator(config)
    c1_tool = generator._enforce_filter_limit(c1_tool)
    c2_tool = generator._enforce_filter_limit(c2_tool)

    if random.random() < 0.5:
        c1_long_exit_cond = GeneticUtils.copy_conditions(parent1.long_exit_conditions)
        c2_long_exit_cond = GeneticUtils.copy_conditions(parent2.long_exit_conditions)
    else:
        c1_long_exit_cond = GeneticUtils.copy_conditions(parent2.long_exit_conditions)
        c2_long_exit_cond = GeneticUtils.copy_conditions(parent1.long_exit_conditions)

    if random.random() < 0.5:
        c1_short_exit_cond = GeneticUtils.copy_conditions(parent1.short_exit_conditions)
        c2_short_exit_cond = GeneticUtils.copy_conditions(parent2.short_exit_conditions)
    else:
        c1_short_exit_cond = GeneticUtils.copy_conditions(parent2.short_exit_conditions)
        c2_short_exit_cond = GeneticUtils.copy_conditions(parent1.short_exit_conditions)

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
        exit_gene=c1_exit,
        long_exit_conditions=c1_long_exit_cond,
        short_exit_conditions=c1_short_exit_cond,
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
        exit_gene=c2_exit,
        long_exit_conditions=c1_long_exit_cond,
        short_exit_conditions=c2_short_exit_cond,
        tool_genes=c2_tool,
        metadata=c2_meta,
    )

    return child1, child2
