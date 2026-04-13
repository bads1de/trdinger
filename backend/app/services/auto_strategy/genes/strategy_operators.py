"""
StrategyGene の遺伝的演算ロジック。

遺伝的アルゴリズムにおける交叉（crossover）、突然変異（mutation）、
適応的突然変異（adaptive mutation）などの演算を提供します。
"""

from __future__ import annotations

import inspect
import logging
import random
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .conditions import ConditionGroup
from .entry import EntryGene, create_random_entry_gene
from .genetic_utils import GeneticUtils
from .position_sizing import (
    PositionSizingGene,
    create_random_position_sizing_gene,
)
from .strategy import StrategyGene
from .tpsl import TPSLGene, create_random_tpsl_gene

logger = logging.getLogger(__name__)


_SUB_GENE_MUTATION_RULES = {
    TPSLGene: (
        create_random_tpsl_gene,
        "tpsl_gene_creation_probability_multiplier",
    ),
    PositionSizingGene: (
        create_random_position_sizing_gene,
        "position_sizing_gene_creation_probability_multiplier",
    ),
    EntryGene: (
        create_random_entry_gene,
        "entry_gene_creation_probability_multiplier",
    ),
}

_MUTATION_CONFIG_CREATION_ATTR_MAP = {
    "tpsl_gene_creation_probability_multiplier": "tpsl_gene_creation_multiplier",
    "position_sizing_gene_creation_probability_multiplier": (
        "position_sizing_gene_creation_multiplier"
    ),
    "entry_gene_creation_probability_multiplier": "entry_gene_creation_multiplier",
}


def _get_creation_probability_multiplier(config: Any, attr_name: str) -> float:
    """突然変異で使用する生成確率倍率を安全に取得する。

    設定オブジェクトから指定された属性名の確率倍率を取得します。
    属性が存在しない場合はデフォルト値0.0を返します。

    Args:
        config: GA設定オブジェクト。mutation_config属性を持つ。
        attr_name: 取得する確率倍率の属性名。

    Returns:
        float: 確率倍率（0.0以上の値）。属性が存在しない場合は0.0。
    """
    nested_attr_name = _MUTATION_CONFIG_CREATION_ATTR_MAP.get(attr_name)
    mutation_config = getattr(config, "mutation_config", None)

    try:
        return float(getattr(config, attr_name) or 0.0)
    except (AttributeError, TypeError, ValueError):
        pass

    if nested_attr_name is not None and mutation_config is not None:
        try:
            value = getattr(mutation_config, nested_attr_name)
            return float(value or 0.0)
        except (AttributeError, TypeError, ValueError):
            pass

    return 0.0


def _create_sub_gene(creator_func: Callable[..., object], config: object) -> object:
    """生成関数のシグネチャに応じて設定を渡し分ける。

    サブ遺伝子（TPSL、PositionSizing、Entryなど）の生成関数を呼び出し、
    必要に応じて設定オブジェクトを引数として渡します。

    Args:
        creator_func: サブ遺伝子を生成する関数。
        config: GA設定オブジェクト。

    Returns:
        生成されたサブ遺伝子インスタンス。
    """
    try:
        signature = inspect.signature(creator_func)
    except (TypeError, ValueError):
        return creator_func()

    if len(signature.parameters) == 0:
        return creator_func()
    return creator_func(config)


def _iter_mutable_sub_gene_specs(config: Any) -> list[tuple[str, Any, float]]:
    """StrategyGene の定義を基準に、突然変異対象のサブ遺伝子を列挙する。

    StrategyGeneが持つサブ遺伝子フィールド（TPSL、PositionSizing、Entryなど）
    を走査し、突然変異の対象となるフィールド名、生成関数、確率倍率の
    タプルを生成します。

    Args:
        config: GA設定オブジェクト。各サブ遺伝子の生成確率倍率を含む。

    Returns:
        list[tuple[str, Any, float]]: （フィールド名、生成関数、確率倍率）のリスト。
    """
    class_map = StrategyGene.sub_gene_class_map()
    specs: list[tuple[str, Any, float]] = []

    for field_name in StrategyGene.sub_gene_field_names():
        gene_class = class_map.get(field_name)
        rule = _SUB_GENE_MUTATION_RULES.get(gene_class)
        if rule is None:
            continue

        creator_func, creation_prob_attr = rule
        specs.append(
            (
                field_name,
                creator_func,
                _get_creation_probability_multiplier(config, creation_prob_attr),
            )
        )

    return specs


def mutate_indicators(mutated, mutation_rate: float, config: Any) -> None:
    """指標遺伝子の突然変異処理。

    指標の期間パラメータの変動、指標の追加・削除を確率的に実行します。

    Args:
        mutated: 突然変異を適用する戦略遺伝子クローン。
        mutation_rate: 突然変異確率（0.0〜1.0）。
        config: GA設定オブジェクト。mutation_config属性に突然変異パラメータを持つ。
    """
    min_multiplier, max_multiplier = config.mutation_config.indicator_param_range

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
                            param_value * random.uniform(min_multiplier, max_multiplier)
                        )

    if random.random() < mutation_rate * config.mutation_config.indicator_add_delete_probability:
        max_indicators = config.max_indicators
        if (
            len(mutated.indicators) < max_indicators
            and random.random() < config.mutation_config.indicator_add_vs_delete_probability
        ):
            from .indicator import generate_random_indicators
            from .validator import GeneValidator

            validator = GeneValidator()
            new_indicators = generate_random_indicators(config)
            allowed_indicators = {indicator.type for indicator in new_indicators}
            valid_new_indicators = [
                indicator
                for indicator in new_indicators
                if validator.validate_indicator_gene_for_generation(
                    indicator,
                    indicator_universe_mode=getattr(
                        config, "indicator_universe_mode", "curated"
                    ),
                    allowed_indicators=allowed_indicators,
                )
            ]
            if valid_new_indicators:
                mutated.indicators.append(random.choice(valid_new_indicators))
        elif len(mutated.indicators) > config.min_indicators and random.random() < (
            1 - config.mutation_config.indicator_add_vs_delete_probability
        ):
            mutated.indicators.pop(random.randint(0, len(mutated.indicators) - 1))


def mutate_conditions(mutated, mutation_rate: float, config: Any) -> None:
    """条件の突然変異処理。

    比較演算子の反転、閾値の微調整を確率的に実行します。
    ConditionGroupの演算子（AND/OR）の切り替えも含みます。

    Args:
        mutated: 突然変異を適用する戦略遺伝子クローン。
        mutation_rate: 突然変異確率（0.0〜1.0）。
        config: GA設定オブジェクト。mutation_config属性に突然変異パラメータを持つ。
    """

    def mutate_item(condition):
        if isinstance(condition, ConditionGroup):
            if random.random() < config.mutation_config.condition_operator_switch_probability:
                condition.operator = "AND" if condition.operator == "OR" else "OR"
            elif condition.conditions:
                idx = random.randint(0, len(condition.conditions) - 1)
                mutate_item(condition.conditions[idx])
        elif hasattr(condition, "operator"):
            # Condition オブジェクトの演算子を変異
            condition.operator = random.choice(config.mutation_config.valid_condition_operators)
        else:
            # 予期しない型はスキップ（AttributeError 防止）
            logger.debug(f"条件変異をスキップ: 未知の型 {type(condition).__name__}")

    mutation_threshold = mutation_rate * config.mutation_config.condition_change_multiplier

    def maybe_mutate_branch(conditions):
        if conditions and random.random() < config.mutation_config.condition_selection_probability:
            idx = random.randint(0, len(conditions) - 1)
            mutate_item(conditions[idx])

    for conditions in (mutated.long_entry_conditions, mutated.short_entry_conditions):
        if random.random() < mutation_threshold:
            maybe_mutate_branch(conditions)


def mutate_indicators_batch(
    individuals: List[Any], mutation_rate: float, config: Any
) -> List[Any]:
    """指標遺伝子の突然変異処理（バッチ版）。

    複数の個体に対して一括して指標の突然変異を適用します。

    Args:
        individuals: 突然変異を適用する個体のリスト。
        mutation_rate: 突然変異確率（0.0〜1.0）。
        config: GA設定オブジェクト。

    Returns:
        List[Any]: 突然変異が適用された新しい個体のリスト。
    """
    results: List[Any] = []
    for individual in individuals:
        mutated = individual.clone() if hasattr(individual, "clone") else individual
        mutate_indicators(mutated, mutation_rate, config)
        results.append(mutated)
    return results


def mutate_conditions_batch(
    individuals: List[Any], mutation_rate: float, config: Any
) -> List[Any]:
    """条件の突然変異処理（バッチ版）。

    複数の個体に対して一括して条件の突然変異を適用します。

    Args:
        individuals: 突然変異を適用する個体のリスト。
        mutation_rate: 突然変異確率（0.0〜1.0）。
        config: GA設定オブジェクト。

    Returns:
        List[Any]: 突然変異が適用された新しい個体のリスト。
    """
    results: List[Any] = []
    for individual in individuals:
        mutated = individual.clone() if hasattr(individual, "clone") else individual
        mutate_conditions(mutated, mutation_rate, config)
        results.append(mutated)
    return results


def mutate_strategy_gene(gene, config: Any, mutation_rate: float = 0.1):
    """
    戦略遺伝子の「突然変異（Mutation）」を実行し、新しい個体を生成します。

    この関数は、個体の多様性を維持し、局所最適解に陥るのを防ぐために、
    確率的に以下の部位を書き換えます：
    1. **指標（Indicators）**: 期間パラメータの変動、指標の追加・削除。
    2. **取引条件（Conditions）**: 比較演算子の反転、閾値の微調整。
    3. **リスク管理**: ポジションサイズ等の数値パラメータの変動。
    4. **サブ遺伝子（TPSL, PositionSizing, Entry）**:
       - 既存のサブ遺伝子のパラメータを微調整（ガウスノイズ等）。
       - 低い確率で全く新しいランダムなサブ遺伝子に置き換え。

    Args:
        gene (StrategyGene): 突然変異の対象となる親個体。
        config (Any): 突然変異率やパラメータ変動範囲を含むGA設定。
        mutation_rate (float): 基本となる突然変異確率。

    Returns:
        StrategyGene: 突然変異が適用された後の新しい子個体。
    """
    try:
        mutated = gene.clone()

        mutate_indicators(mutated, mutation_rate, config)
        mutate_conditions(mutated, mutation_rate, config)

        min_risk_multiplier, max_risk_multiplier = config.mutation_config.risk_param_range
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

        for field_name, creator_func, creation_prob_mult in _iter_mutable_sub_gene_specs(
            config
        ):
            sub_gene = getattr(mutated, field_name)
            if sub_gene:
                if random.random() < mutation_rate:
                    if isinstance(sub_gene, PositionSizingGene):
                        setattr(
                            mutated,
                            field_name,
                            sub_gene.mutate(mutation_rate, config=config),
                        )
                    else:
                        setattr(mutated, field_name, sub_gene.mutate(mutation_rate))
            elif random.random() < mutation_rate * creation_prob_mult:
                setattr(mutated, field_name, _create_sub_gene(creator_func, config))

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


def mutate_strategy_gene_batch(
    individuals: List[Any], config: Any, mutation_rate: float = 0.1
) -> List[Any]:
    """StrategyGene の突然変異をバッチで実行する。

    複数の個体に対して指標、条件、その他の突然変異を一括適用します。

    Args:
        individuals: 突然変異を適用する個体のリスト。
        config: GA設定オブジェクト。
        mutation_rate: 基本となる突然変異確率（0.0〜1.0）。

    Returns:
        List[Any]: 突然変異が適用された新しい個体のリスト。
    """
    mutated = mutate_indicators_batch(individuals, mutation_rate, config)
    mutated = mutate_conditions_batch(mutated, mutation_rate, config)
    return [
        mutate_strategy_gene(individual, config, mutation_rate=mutation_rate)
        for individual in mutated
    ]


def adaptive_mutate_strategy_gene(
    gene,
    population,
    config: Any,
    base_mutation_rate: float = 0.1,
):
    """集団分散に基づく適応的突然変異。

    集団のフィットネス分散に応じて突然変異率を動的に調整します。
    分散が低い（集団が収束している）場合は突然変異率を上げ、
    分散が高い場合は下げることで、探索と活用のバランスを取ります。

    Args:
        gene: 突然変異を適用する個体。
        population: 現在の集団全体。フィットネス分散の計算に使用。
        config: GA設定オブジェクト。mutation_config属性に適応的突然変異パラメータを持つ。
        base_mutation_rate: 基本となる突然変異確率（0.0〜1.0）。

    Returns:
        突然変異が適用された新しい個体。
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
            variance_threshold = config.mutation_config.adaptive_variance_threshold

            if variance > variance_threshold:
                adaptive_rate = (
                    base_mutation_rate
                    * config.mutation_config.adaptive_decrease_multiplier
                )
            else:
                adaptive_rate = (
                    base_mutation_rate
                    * config.mutation_config.adaptive_increase_multiplier
                )

            adaptive_rate = max(0.01, min(1.0, adaptive_rate))

        mutated = mutate_strategy_gene(gene, config, mutation_rate=adaptive_rate)
        mutated.metadata["adaptive_mutation_rate"] = adaptive_rate
        return mutated

    except Exception as e:
        logger.error(f"適応的戦略遺伝子突然変異エラー: {e}")
        return mutate_strategy_gene(gene, config, mutation_rate=base_mutation_rate)


def crossover_tpsl_genes(
    parent1_tpsl: Optional[TPSLGene],
    parent2_tpsl: Optional[TPSLGene],
) -> Tuple[Optional[TPSLGene], Optional[TPSLGene]]:
    """TPSL遺伝子の交叉を実行する。

    2つの親TPSL遺伝子から、新しい2つの子TPSL遺伝子を生成します。
    両方の親がNoneの場合は(None, None)を返します。

    Args:
        parent1_tpsl: 1番目の親TPSL遺伝子。
        parent2_tpsl: 2番目の親TPSL遺伝子。

    Returns:
        Tuple[Optional[TPSLGene], Optional[TPSLGene]]: 交叉された2つの子TPSL遺伝子。
    """
    return GeneticUtils.crossover_optional_gene(parent1_tpsl, parent2_tpsl, TPSLGene)


def crossover_position_sizing_genes(
    parent1_ps: Optional[PositionSizingGene],
    parent2_ps: Optional[PositionSizingGene],
) -> Tuple[Optional[PositionSizingGene], Optional[PositionSizingGene]]:
    """ポジションサイジング遺伝子の交叉を実行する。

    2つの親ポジションサイジング遺伝子から、新しい2つの子を生成します。

    Args:
        parent1_ps: 1番目の親ポジションサイジング遺伝子。
        parent2_ps: 2番目の親ポジションサイジング遺伝子。

    Returns:
        Tuple[Optional[PositionSizingGene], Optional[PositionSizingGene]]: 交叉された2つの子遺伝子。
    """
    return GeneticUtils.crossover_optional_gene(
        parent1_ps, parent2_ps, PositionSizingGene
    )


def crossover_entry_genes(
    parent1_entry: Optional[EntryGene],
    parent2_entry: Optional[EntryGene],
) -> Tuple[Optional[EntryGene], Optional[EntryGene]]:
    """エントリー遺伝子の交叉を実行する。

    2つの親エントリー遺伝子から、新しい2つの子エントリー遺伝子を生成します。

    Args:
        parent1_entry: 1番目の親エントリー遺伝子。
        parent2_entry: 2番目の親エントリー遺伝子。

    Returns:
        Tuple[Optional[EntryGene], Optional[EntryGene]]: 交叉された2つの子エントリー遺伝子。
    """
    return GeneticUtils.crossover_optional_gene(parent1_entry, parent2_entry, EntryGene)


def crossover_strategy_genes(
    strategy_gene_class,
    parent1,
    parent2,
    config: Any,
    crossover_type: str = "uniform",
):
    """
    2つの親個体（StrategyGene）から、新しい2つの子個体を「交叉（Crossover）」により生成します。

    交叉手法：
    - **uniform** (デフォルト): 各属性（指標リスト、条件リスト、TP/SL設定等）ごとに独立して、
      `crossover_field_selection_probability`（通常0.5）の確率で親1または親2から遺伝子を引き継ぎます。
    - **single_point**: 属性リストの中間に一点の切断点を設け、それ以降の属性を親同士で入れ替えます。

    Args:
        strategy_gene_class: 生成する個体のクラス（通常は StrategyGene）。
        parent1: 1番目の親個体。
        parent2: 2番目の親個体。
        config: 交叉確率やフィールド選択確率を含むGA設定。
        crossover_type: 使用する交叉アルゴリズムの識別子。

    Returns:
        Tuple[StrategyGene, StrategyGene]: 属性がシャッフルされた2つの新しい子個体のペア。
    """
    try:
        if crossover_type == "uniform":
            return uniform_crossover(strategy_gene_class, parent1, parent2, config)
        return single_point_crossover(strategy_gene_class, parent1, parent2, config)
    except Exception as e:
        logger.error(f"戦略遺伝子交叉エラー: {e}")
        return parent1, parent2


def crossover_strategy_genes_batch(
    individuals: List[Any], config: Any, crossover_rate: float = 0.8
) -> List[Tuple[Any, Any]]:
    """StrategyGene の交叉をバッチで実行する。

    複数の個体ペアに対して一括して交叉を適用します。
    個体を2つずつペアにして、crossover_rateの確率で交叉を実行します。

    Args:
        individuals: 交叉を適用する個体のリスト。
        config: GA設定オブジェクト。
        crossover_rate: 交叉を実行する確率（0.0〜1.0）。

    Returns:
        List[Tuple[Any, Any]]: 交叉された子個体のペアのリスト。
    """
    results: List[Tuple[Any, Any]] = []

    for i in range(0, len(individuals) - 1, 2):
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

    return results


def uniform_crossover(strategy_gene_class, parent1, parent2, config: Any):
    """ユニフォーム交叉（一様交叉）。

    各属性（フィールド）ごとに独立して、指定された確率で親1または親2から
    遺伝子を引き継ぐ2つの子個体を生成します。

    Args:
        strategy_gene_class: 生成する個体のクラス（通常は StrategyGene）。
        parent1: 1番目の親個体。
        parent2: 2番目の親個体。
        config: GA設定オブジェクト。mutation_config.crossover_field_selection_probability を使用。

    Returns:
        Tuple[StrategyGene, StrategyGene]: 属性がランダムにシャッフルされた2つの新しい子個体。
    """
    selection_prob = config.mutation_config.crossover_field_selection_probability

    child1_params: Dict[str, Any] = {"id": str(uuid.uuid4())}
    child2_params: Dict[str, Any] = {"id": str(uuid.uuid4())}

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
    """一点交叉。

    指標リストの中間に一点の切断点を設け、それ以降の属性を親同士で
    入れ替えることで2つの子個体を生成します。

    Args:
        strategy_gene_class: 生成する個体のクラス（通常は StrategyGene）。
        parent1: 1番目の親個体。
        parent2: 2番目の親個体。
        config: GA設定オブジェクト。max_indicators属性を使用。

    Returns:
        Tuple[StrategyGene, StrategyGene]: 指標リストが分割されて組み換えられた2つの新しい子個体。
    """
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

    c1_tool = (
        GeneticUtils.copy_tool_genes(parent1.tool_genes)
        if random.random() < 0.5
        else GeneticUtils.copy_tool_genes(parent2.tool_genes)
    )
    c2_tool = (
        GeneticUtils.copy_tool_genes(parent2.tool_genes)
        if random.random() < 0.5
        else GeneticUtils.copy_tool_genes(parent1.tool_genes)
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
