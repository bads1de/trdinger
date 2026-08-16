"""
StrategyGene の突然変異（mutation）演算ロジック。

遺伝的アルゴリズムにおける指標・条件・リスクパラメータ・サブ遺伝子の
突然変異処理を提供します。
"""

import contextlib
import inspect
import logging
import random
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from ...config.ga_config import GAConfig
from ..conditions import Condition, ConditionGroup
from ..entry import EntryGene, create_random_entry_gene
from ..exit import ExitGene, create_random_exit_gene
from ..position_sizing import (
    PositionSizingGene,
    create_random_position_sizing_gene,
)
from ..strategy import StrategyGene
from ..tpsl import TPSLGene, create_random_tpsl_gene

logger = logging.getLogger(__name__)


_SUB_GENE_MUTATION_RULES = {
    TPSLGene: (
        create_random_tpsl_gene,
        "tpsl_gene_creation_multiplier",
    ),
    PositionSizingGene: (
        create_random_position_sizing_gene,
        "position_sizing_gene_creation_multiplier",
    ),
    EntryGene: (
        create_random_entry_gene,
        "entry_gene_creation_multiplier",
    ),
    ExitGene: (
        create_random_exit_gene,
        "exit_gene_creation_multiplier",
    ),
}


def _create_sub_gene(creator_func: Callable[..., object], config: object) -> object:
    """生成関数のシグネチャに応じて設定を渡し分ける。"""
    try:
        signature = inspect.signature(creator_func)
    except (TypeError, ValueError):
        return creator_func()

    if len(signature.parameters) == 0:
        return creator_func()
    return creator_func(config)


def _iter_mutable_sub_gene_specs(
    config: GAConfig,
) -> list[tuple[str, object, float]]:
    """StrategyGene の定義を基準に、突然変異対象のサブ遺伝子を列挙する。"""
    class_map = StrategyGene.sub_gene_class_map()
    specs: list[tuple[str, object, float]] = []

    for field_name in StrategyGene.sub_gene_field_names():
        gene_class = class_map.get(field_name)
        rule = _SUB_GENE_MUTATION_RULES.get(gene_class)  # type: ignore[arg-type]

        if rule is None:
            continue

        creator_func, creation_prob_attr = rule
        try:
            creation_prob_mult = float(
                getattr(config.mutation_config, creation_prob_attr) or 0.0
            )
        except (AttributeError, TypeError, ValueError):
            creation_prob_mult = 0.0

        specs.append((field_name, creator_func, creation_prob_mult))

    return specs


def mutate_indicators(
    mutated: StrategyGene, mutation_rate: float, config: GAConfig
) -> None:
    """指標遺伝子の突然変異処理。"""
    min_multiplier, max_multiplier = config.mutation_config.indicator_param_range

    integer_param_names = {
        "period",
        "signal_period",
        "lookback",
        "length",
        "fast_period",
        "slow_period",
        "signal",
        "roc_period",
        "atr_period",
        "std_dev_period",
        "mom_period",
        "cci_period",
        "willr_period",
        "stoch_k_period",
        "stoch_d_period",
        "stoch_slowk",
        "stoch_slowd",
        "obv_period",
        "ad_period",
        "adx_period",
        "aroon_period",
        "bop_period",
    }

    for i, indicator in enumerate(mutated.indicators):
        if random.random() < mutation_rate:
            for param_name, param_value in indicator.parameters.items():
                if (
                    isinstance(param_value, (int, float))
                    and random.random() < mutation_rate
                ):
                    was_integer = isinstance(param_value, int)

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
                    elif was_integer or param_name in integer_param_names:
                        new_value = int(
                            param_value * random.uniform(min_multiplier, max_multiplier)
                        )
                        mutated.indicators[i].parameters[param_name] = max(1, new_value)
                    else:
                        mutated.indicators[i].parameters[param_name] = (
                            param_value * random.uniform(min_multiplier, max_multiplier)
                        )

    if (
        random.random()
        < mutation_rate * config.mutation_config.indicator_add_delete_probability
    ):
        max_indicators = config.max_indicators
        if (
            len(mutated.indicators) < max_indicators
            and random.random()
            < config.mutation_config.indicator_add_vs_delete_probability
        ):
            from ..indicator import generate_random_indicators
            from ..validator import GeneValidator

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


def mutate_conditions(
    mutated: StrategyGene, mutation_rate: float, config: GAConfig
) -> None:
    """条件の突然変異処理。"""

    def mutate_item(condition: Condition | ConditionGroup) -> None:
        if isinstance(condition, ConditionGroup):
            if (
                random.random()
                < config.mutation_config.condition_operator_switch_probability
            ):
                condition.operator = "AND" if condition.operator == "OR" else "OR"
            elif condition.conditions:
                idx = random.randint(0, len(condition.conditions) - 1)
                mutate_item(condition.conditions[idx])
        else:
            try:
                condition.operator = random.choice(
                    config.mutation_config.valid_condition_operators
                )
            except AttributeError:
                logger.debug(f"条件変異をスキップ: 未知の型 {type(condition).__name__}")

    mutation_threshold = (
        mutation_rate * config.mutation_config.condition_change_multiplier
    )

    def maybe_mutate_branch(conditions: list[Condition | ConditionGroup]) -> None:
        if (
            conditions
            and random.random() < config.mutation_config.condition_selection_probability
        ):
            idx = random.randint(0, len(conditions) - 1)
            mutate_item(conditions[idx])

    for conditions in (
        mutated.long_entry_conditions,
        mutated.short_entry_conditions,
        mutated.long_exit_conditions,
        mutated.short_exit_conditions,
    ):
        if random.random() < mutation_threshold:
            maybe_mutate_branch(conditions)


def mutate_strategy_gene(
    gene: StrategyGene, config: GAConfig, mutation_rate: float = 0.1
) -> StrategyGene:
    """戦略遺伝子の「突然変異（Mutation）」を実行し、新しい個体を生成する。"""
    try:
        mutated = gene.clone()

        with contextlib.suppress(Exception):
            mutate_indicators(mutated, mutation_rate, config)

        with contextlib.suppress(Exception):
            mutate_conditions(mutated, mutation_rate, config)

        try:
            min_risk_multiplier, max_risk_multiplier = (
                config.mutation_config.risk_param_range
            )
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
        except Exception:
            pass

        try:
            for (
                field_name,
                creator_func,
                creation_prob_mult,
            ) in _iter_mutable_sub_gene_specs(config):
                sub_gene = getattr(mutated, field_name)
                if sub_gene:
                    if random.random() < mutation_rate:
                        setattr(mutated, field_name, sub_gene.mutate(mutation_rate))
                elif random.random() < mutation_rate * creation_prob_mult:
                    setattr(
                        mutated,
                        field_name,
                        _create_sub_gene(creator_func, config),  # type: ignore[arg-type]
                    )
        except Exception:
            pass

        try:
            if mutated.tool_genes:
                from ...tools import tool_registry

                for tool_gene in mutated.tool_genes:
                    if random.random() < mutation_rate:
                        if random.random() < 0.2:
                            tool_gene.enabled = not tool_gene.enabled

                        tool = tool_registry.get(tool_gene.tool_name)
                        if tool:
                            tool_gene.params = tool.mutate_params(tool_gene.params)

                # フィルター数制限を強制
                from ...generators.random_gene_generator import RandomGeneGenerator

                mutated.tool_genes = RandomGeneGenerator.enforce_filter_limit(
                    config, mutated.tool_genes
                )
        except Exception:
            pass

        mutated.metadata["mutated"] = True
        mutated.metadata["mutation_rate"] = mutation_rate
        mutated.id = str(uuid.uuid4())

        # 非価格指標（OI/FR/LSR由来）の最低数を確保
        mutated.repair_non_price_indicators(config)

        # 指標の削除・交換で条件参照が切れるため、整合性を修復する
        mutated.repair_condition_references()

        # 旧シード由来の価格×非価格スケール比較（恒真/恒偽）を閾値比較へ書き直す
        mutated.repair_condition_scales()

        return mutated

    except Exception as e:
        logger.error(f"戦略遺伝子突然変異エラー: {e}")
        try:
            mutated = gene.clone()
        except AttributeError:
            mutated = gene
        try:
            del mutated.fitness.values  # type: ignore[attr-defined]
        except AttributeError:
            pass
        return mutated


def adaptive_mutate_strategy_gene(
    gene: StrategyGene,
    population: list[Any],
    config: GAConfig,
    base_mutation_rate: float = 0.1,
) -> StrategyGene:
    """集団分散に基づく適応的突然変異。"""
    try:
        fitnesses = []
        for ind in population:
            try:
                if ind.fitness and ind.fitness.values:
                    fitnesses.append(ind.fitness.values[0])
            except AttributeError:
                pass

        # ±inf（制約違反ペナルティ）の fitness は分散計算の対象外とする。
        # そのまま np.var に渡すと RuntimeWarning（invalid value encountered
        # in subtract）を引き起こすため除外する。
        finite_fitnesses = [f for f in fitnesses if np.isfinite(f)]
        if not finite_fitnesses:
            adaptive_rate = base_mutation_rate
        else:
            variance = float(np.var(finite_fitnesses))
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
