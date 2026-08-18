"""
StrategyGene の交叉（crossover）演算ロジック。

遺伝的アルゴリズムにおけるユニフォーム交叉・一点交叉を提供します。
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import TYPE_CHECKING, Any

from ..entry import EntryGene
from ..exit import ExitGene
from ..genetic_utils import GeneticUtils
from ..position_sizing import PositionSizingGene
from ..strategy import StrategyGene
from ..tpsl import TPSLGene

if TYPE_CHECKING:
    from ...config.ga_config import GAConfig

logger = logging.getLogger(__name__)

WEEKEND_FILTER_NAME = "weekend_filter"


def _ensure_weekend_filter(tool_genes: list[Any]) -> list[Any]:
    """weekend_filter が常時ONで存在することを保証する。"""
    from ..tool import ToolGene

    existing = next(
        (t for t in tool_genes if getattr(t, "tool_name", None) == WEEKEND_FILTER_NAME),
        None,
    )
    if existing is not None:
        existing.enabled = True
    else:
        tool_genes.append(
            ToolGene(tool_name=WEEKEND_FILTER_NAME, enabled=True, params={})
        )
    return tool_genes


def _crossover_optional(
    gene_class: type[Any],
    parent1_gene: Any | None,
    parent2_gene: Any | None,
) -> tuple[Any | None, Any | None]:
    """サブ遺伝子の交叉を実行する（None 許容）。"""
    return GeneticUtils.crossover_optional_gene(parent1_gene, parent2_gene, gene_class)


def crossover_strategy_genes(
    strategy_gene_class: type[StrategyGene],
    parent1: StrategyGene,
    parent2: StrategyGene,
    config: GAConfig,
    crossover_type: str = "uniform",
) -> tuple[StrategyGene, StrategyGene]:
    """2つの親個体から新しい2つの子個体を交叉により生成する。"""
    try:
        if crossover_type == "uniform":
            return uniform_crossover(strategy_gene_class, parent1, parent2, config)
        return single_point_crossover(strategy_gene_class, parent1, parent2, config)
    except Exception as e:
        logger.error(f"戦略遺伝子交叉エラー: {e}")
        try:
            child1 = parent1.clone()
        except AttributeError:
            child1 = parent1
        try:
            child2 = parent2.clone()
        except AttributeError:
            child2 = parent2
        for child in (child1, child2):
            try:
                del child.fitness.values  # type: ignore[attr-defined]
            except AttributeError:
                pass
        return child1, child2


def uniform_crossover(
    strategy_gene_class: type[StrategyGene],
    parent1: StrategyGene,
    parent2: StrategyGene,
    config: GAConfig,
) -> tuple[StrategyGene, StrategyGene]:
    """ユニフォーム交叉（一様交叉）。"""
    selection_prob = config.mutation_config.crossover_field_selection_probability

    child1_params: dict[str, Any] = {"id": str(uuid.uuid4())}
    child2_params: dict[str, Any] = {"id": str(uuid.uuid4())}

    try:
        fields = strategy_gene_class.crossover_field_names()
    except AttributeError:
        fields = ()

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

    # weekend_filter 常時ONを保証
    for params in (child1_params, child2_params):
        tgs: list[Any] = params.get("tool_genes") or []
        params["tool_genes"] = _ensure_weekend_filter(list(tgs))

    child1 = strategy_gene_class(**child1_params)
    child2 = strategy_gene_class(**child2_params)

    # フィールド単位のスワップで条件と指標が別々の親に混ざるため、
    # 参照切れの条件を除去して整合性を保つ
    child1.repair_condition_references()
    child2.repair_condition_references()

    # 旧シード由来の価格×非価格スケール比較（恒真/恒偽）を閾値比較へ書き直す
    child1.repair_condition_scales()
    child2.repair_condition_scales()

    return child1, child2


def single_point_crossover(
    strategy_gene_class: type[StrategyGene],
    parent1: StrategyGene,
    parent2: StrategyGene,
    config: GAConfig,
) -> tuple[StrategyGene, StrategyGene]:
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

    c1_risk: dict[str, Any] = {}
    c2_risk: dict[str, Any] = {}
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

    c1_tpsl, c2_tpsl = _crossover_optional(
        TPSLGene, parent1.tpsl_gene, parent2.tpsl_gene
    )
    c1_long_tpsl, c2_long_tpsl = _crossover_optional(
        TPSLGene,
        parent1.long_tpsl_gene,
        parent2.long_tpsl_gene,
    )
    c1_short_tpsl, c2_short_tpsl = _crossover_optional(
        TPSLGene,
        parent1.short_tpsl_gene,
        parent2.short_tpsl_gene,
    )
    c1_ps, c2_ps = _crossover_optional(
        PositionSizingGene,
        parent1.position_sizing_gene,
        parent2.position_sizing_gene,
    )
    c1_entry, c2_entry = _crossover_optional(
        EntryGene, parent1.entry_gene, parent2.entry_gene
    )
    c1_long_entry, c2_long_entry = _crossover_optional(
        EntryGene,
        parent1.long_entry_gene,
        parent2.long_entry_gene,
    )
    c1_short_entry, c2_short_entry = _crossover_optional(
        EntryGene,
        parent1.short_entry_gene,
        parent2.short_entry_gene,
    )
    c1_exit, c2_exit = _crossover_optional(
        ExitGene, parent1.exit_gene, parent2.exit_gene
    )

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

    c1_tool = RandomGeneGenerator.enforce_filter_limit(config, c1_tool)
    c2_tool = RandomGeneGenerator.enforce_filter_limit(config, c2_tool)
    c1_tool = _ensure_weekend_filter(c1_tool)
    c2_tool = _ensure_weekend_filter(c2_tool)

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
        long_exit_conditions=c2_long_exit_cond,
        short_exit_conditions=c2_short_exit_cond,
        tool_genes=c2_tool,
        metadata=c2_meta,
    )

    # 指標は両親の混在、条件はどちらか一方の親から丸ごと渡るため、
    # 参照切れの条件を除去して整合性を保つ
    child1.repair_condition_references()
    child2.repair_condition_references()

    # 旧シード由来の価格×非価格スケール比較（恒真/恒偽）を閾値比較へ書き直す
    child1.repair_condition_scales()
    child2.repair_condition_scales()

    return child1, child2
