"""
遺伝的演算子

戦略遺伝子の交叉・突然変異ロジックを担当します。
"""

import copy
import logging
import random
import uuid
from dataclasses import asdict
from typing import Union

import numpy as np

from ..genes import (
    StrategyGene,
    create_random_tpsl_gene,
    crossover_position_sizing_genes,
    crossover_tpsl_genes,
    mutate_position_sizing_gene,
    mutate_tpsl_gene,
)

logger = logging.getLogger(__name__)


def _convert_to_strategy_gene(individual_or_gene) -> StrategyGene:
    """
    IndividualオブジェクトまたはStrategyGeneオブジェクトをStrategyGeneに変換

    Args:
        individual_or_gene: DEAPのIndividualオブジェクトまたはStrategyGeneオブジェクト

    Returns:
        StrategyGeneオブジェクト
    """
    # StrategyGeneオブジェクトの場合はそのまま返す
    if hasattr(individual_or_gene, "indicators"):
        return individual_or_gene

    # DEAPのIndividualオブジェクト（リスト）の場合はデコード
    if isinstance(individual_or_gene, list):
        try:
            from ..serializers.gene_serialization import GeneSerializer

            gene_serializer = GeneSerializer()
            return gene_serializer.decode_list_to_strategy_gene(
                individual_or_gene, StrategyGene
            )

        except Exception as e:
            logger.error(f"Individual→StrategyGene変換エラー: {e}")
            raise

    raise TypeError(f"サポートされていない型です: {type(individual_or_gene)}")


def _crossover_tpsl_genes(parent1_tpsl, parent2_tpsl):
    """
    TP/SL遺伝子の交叉を処理

    Args:
        parent1_tpsl: 親1のTP/SL遺伝子
        parent2_tpsl: 親2のTP/SL遺伝子

    Returns:
        子1と子2のTP/SL遺伝子のタプル
    """
    if parent1_tpsl and parent2_tpsl:
        return crossover_tpsl_genes(parent1_tpsl, parent2_tpsl)
    elif parent1_tpsl:
        return parent1_tpsl, parent1_tpsl  # コピー
    elif parent2_tpsl:
        return parent2_tpsl, parent2_tpsl  # コピー
    else:
        return None, None


def _crossover_position_sizing_genes(parent1_ps, parent2_ps):
    """
    ポジションサイジング遺伝子の交叉を処理

    Args:
        parent1_ps: 親1のポジションサイジング遺伝子
        parent2_ps: 親2のポジションサイジング遺伝子

    Returns:
        子1と子2のポジションサイジング遺伝子のタプル
    """
    if parent1_ps and parent2_ps:
        return crossover_position_sizing_genes(parent1_ps, parent2_ps)
    elif parent1_ps:
        return parent1_ps, copy.deepcopy(parent1_ps)
    elif parent2_ps:
        return parent2_ps, copy.deepcopy(parent2_ps)
    else:
        return None, None


def _mutate_indicators(mutated, gene, mutation_rate, config):
    """
    指標の突然変異を処理

    Args:
        mutated: 突然変異対象のStrategyGene
        gene: 元のStrategyGene
        mutation_rate: 突然変異率
        config: GAConfigオブジェクト
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
                        # 期間パラメータの上限下限をConfigから取得
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

    # 指標の追加・削除
    if random.random() < mutation_rate * config.indicator_add_delete_probability:
        max_indicators = config.max_indicators

        # 指標追加の確率
        if (
            len(mutated.indicators) < max_indicators
            and random.random() < config.indicator_add_vs_delete_probability
        ):
            # 新しい指標を追加
            from ..generators.random_gene_generator import RandomGeneGenerator

            generator = RandomGeneGenerator(config)  # configを渡す
            new_indicators = generator.indicator_generator.generate_random_indicators()
            if new_indicators:
                mutated.indicators.append(random.choice(new_indicators))
        # 指標削除の確率
        elif len(mutated.indicators) > config.min_indicators and random.random() < (
            1 - config.indicator_add_vs_delete_probability
        ):
            # 指標を削除
            mutated.indicators.pop(random.randint(0, len(mutated.indicators) - 1))


def _mutate_condition_item(condition, mutation_rate, config):
    """
    個々の条件（ConditionまたはConditionGroup）を変異させる再帰関数
    """
    import random

    from ..genes import ConditionGroup

    if isinstance(condition, ConditionGroup):
        # ConditionGroupの場合
        if random.random() < config.condition_operator_switch_probability:
            # グループ自体のオペレータを変異 (AND <-> OR)
            condition.operator = "AND" if condition.operator == "OR" else "OR"
        else:
            # 内部の条件を再帰的に変異
            if condition.conditions:
                idx = random.randint(0, len(condition.conditions) - 1)
                _mutate_condition_item(condition.conditions[idx], mutation_rate, config)
    else:
        # 通常のConditionの場合
        condition.operator = random.choice(config.valid_condition_operators)


def _mutate_conditions(mutated, mutation_rate, config):
    """
    条件の突然変異を処理

    Args:
        mutated: 突然変異対象のStrategyGene
        mutation_rate: 突然変異率
        config: GAConfigオブジェクト
    """
    if random.random() < mutation_rate * config.condition_change_probability_multiplier:
        # ロングエントリー条件の変更
        if (
            mutated.long_entry_conditions
            and random.random() < config.condition_selection_probability
        ):
            condition_idx = random.randint(0, len(mutated.long_entry_conditions) - 1)
            condition = mutated.long_entry_conditions[condition_idx]
            _mutate_condition_item(condition, mutation_rate, config)

    if random.random() < mutation_rate * config.condition_change_probability_multiplier:
        # ショートエントリー条件の変更
        if (
            mutated.short_entry_conditions
            and random.random() < config.condition_selection_probability
        ):
            condition_idx = random.randint(0, len(mutated.short_entry_conditions) - 1)
            condition = mutated.short_entry_conditions[condition_idx]
            _mutate_condition_item(condition, mutation_rate, config)


def _convert_to_individual(strategy_gene: StrategyGene, individual_class=None):
    """
    StrategyGeneオブジェクトをIndividualオブジェクトに変換

    Args:
        strategy_gene: StrategyGeneオブジェクト
        individual_class: DEAPのIndividualクラス（Noneの場合はStrategyGeneを返す）

    Returns:
        IndividualオブジェクトまたはStrategyGeneオブジェクト
    """
    if individual_class is None:
        return strategy_gene

    try:
        # 既にIndividualクラスのインスタンスならそのまま返す
        if isinstance(strategy_gene, individual_class):
            return strategy_gene

        # StrategyGeneのフィールドを展開してIndividualを生成
        return individual_class(**asdict(strategy_gene))
    except Exception as e:
        logger.error(f"StrategyGene→Individual変換エラー: {e}")
        raise


def crossover_strategy_genes_pure(
    parent1: StrategyGene,
    parent2: StrategyGene,
    config,
    crossover_type: str = "uniform",
) -> tuple[StrategyGene, StrategyGene]:
    """
    戦略遺伝子の交叉（純粋版）

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な交叉を実行します。
    StrategyGeneオブジェクトのみを扱い、DEAPライブラリに依存しません。

    Args:
        parent1: 親1の戦略遺伝子（StrategyGeneオブジェクト）
        parent2: 親2の戦略遺伝子（StrategyGeneオブジェクト）
        config: GAConfigオブジェクト
        crossover_type: 交叉タイプ ("single_point" または "uniform")

    Returns:
        交叉後の子1、子2の戦略遺伝子のタプル
    """
    try:
        if crossover_type == "uniform":
            return uniform_crossover(parent1, parent2, config)
        else:
            # 一点交叉（既存ロジック）
            # 指標遺伝子の交叉（単純な一点交叉）
            min_indicators = min(len(parent1.indicators), len(parent2.indicators))
            if min_indicators <= 1:
                # 指標数が1以下の場合は交叉点を0に設定（全体を交換）
                crossover_point = 0
            else:
                crossover_point = random.randint(1, min_indicators)

            child1_indicators = (
                parent1.indicators[:crossover_point]
                + parent2.indicators[crossover_point:]
            )
            child2_indicators = (
                parent2.indicators[:crossover_point]
                + parent1.indicators[crossover_point:]
            )

            # 最大指標数制限
            max_indicators = config.max_indicators

            child1_indicators = child1_indicators[:max_indicators]
            child2_indicators = child2_indicators[:max_indicators]

            # リスク管理設定の交叉（平均値）
            child1_risk = {}
            child2_risk = {}

            all_keys = set(parent1.risk_management.keys()) | set(
                parent2.risk_management.keys()
            )
            for key in all_keys:
                val1 = parent1.risk_management.get(key, 0)
                val2 = parent2.risk_management.get(key, 0)

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    child1_risk[key] = (val1 + val2) / 2
                    child2_risk[key] = (val1 + val2) / 2
                else:
                    child1_risk[key] = val1 if random.random() < 0.5 else val2
                    child2_risk[key] = val2 if random.random() < 0.5 else val1

            # TP/SL遺伝子の交叉
            child1_tpsl, child2_tpsl = _crossover_tpsl_genes(
                parent1.tpsl_gene, parent2.tpsl_gene
            )
            child1_long_tpsl, child2_long_tpsl = _crossover_tpsl_genes(
                parent1.long_tpsl_gene, parent2.long_tpsl_gene
            )
            child1_short_tpsl, child2_short_tpsl = _crossover_tpsl_genes(
                parent1.short_tpsl_gene, parent2.short_tpsl_gene
            )

            # ポジションサイジング遺伝子の交叉
            ps_gene1 = getattr(parent1, "position_sizing_gene", None)
            ps_gene2 = getattr(parent2, "position_sizing_gene", None)
            child1_position_sizing, child2_position_sizing = (
                _crossover_position_sizing_genes(ps_gene1, ps_gene2)
            )

            # メタデータの交叉（共通ユーティリティ使用）
            from ..utils.gene_utils import prepare_crossover_metadata

            child1_metadata, child2_metadata = prepare_crossover_metadata(
                parent1, parent2
            )

            # ロング・ショート条件の交叉
            if random.random() < 0.5:
                child1_long_entry = parent1.long_entry_conditions.copy()
                child2_long_entry = parent2.long_entry_conditions.copy()
            else:
                child1_long_entry = parent2.long_entry_conditions.copy()
                child2_long_entry = parent1.long_entry_conditions.copy()

            if random.random() < 0.5:
                child1_short_entry = parent1.short_entry_conditions.copy()
                child2_short_entry = parent2.short_entry_conditions.copy()
            else:
                child1_short_entry = parent2.short_entry_conditions.copy()
                child2_short_entry = parent1.short_entry_conditions.copy()

            # ツール遺伝子の交叉
            parent1_tool_genes = getattr(parent1, "tool_genes", []) or []
            parent2_tool_genes = getattr(parent2, "tool_genes", []) or []
            if random.random() < 0.5:
                child1_tool_genes = copy.deepcopy(parent1_tool_genes)
                child2_tool_genes = copy.deepcopy(parent2_tool_genes)
            else:
                child1_tool_genes = copy.deepcopy(parent2_tool_genes)
                child2_tool_genes = copy.deepcopy(parent1_tool_genes)

            # 子遺伝子の作成
            child1_strategy = StrategyGene(
                id=str(uuid.uuid4()),
                indicators=child1_indicators,
                long_entry_conditions=child1_long_entry,
                short_entry_conditions=child1_short_entry,
                risk_management=child1_risk,
                tpsl_gene=child1_tpsl,
                long_tpsl_gene=child1_long_tpsl,
                short_tpsl_gene=child1_short_tpsl,
                position_sizing_gene=child1_position_sizing,
                tool_genes=child1_tool_genes,
                metadata=child1_metadata,
            )

            child2_strategy = StrategyGene(
                id=str(uuid.uuid4()),
                indicators=child2_indicators,
                long_entry_conditions=child2_long_entry,
                short_entry_conditions=child2_short_entry,
                risk_management=child2_risk,
                tpsl_gene=child2_tpsl,
                long_tpsl_gene=child2_long_tpsl,
                short_tpsl_gene=child2_short_tpsl,
                position_sizing_gene=child2_position_sizing,
                tool_genes=child2_tool_genes,
                metadata=child2_metadata,
            )

            return child1_strategy, child2_strategy

    except Exception as e:
        logger.error(f"戦略遺伝子交叉エラー: {e}")
        # エラー時は親をそのまま返す
        return parent1, parent2


def uniform_crossover(
    parent1: StrategyGene, parent2: StrategyGene, config
) -> tuple[StrategyGene, StrategyGene]:
    """
    ユニフォーム交叉

    StrategyGeneの各フィールドについて、各遺伝子位置でランダムに親を選択します。
    多様性を高めるために使用されます。

    Args:
        parent1: 親1の戦略遺伝子（StrategyGeneオブジェクト）
        parent2: 親2の戦略遺伝子（StrategyGeneオブジェクト）
        config: GAConfigオブジェクト

    Returns:
        交叉後の子1、子2の戦略遺伝子のタプル
    """
    try:
        selection_prob = config.crossover_field_selection_probability

        # 各フィールドに対してランダム選択
        child1_indicators = (
            parent1.indicators
            if random.random() < selection_prob
            else parent2.indicators
        )
        child2_indicators = (
            parent2.indicators
            if random.random() < selection_prob
            else parent1.indicators
        )

        child1_long_entry_conditions = (
            parent1.long_entry_conditions
            if random.random() < selection_prob
            else parent2.long_entry_conditions
        )
        child2_long_entry_conditions = (
            parent2.long_entry_conditions
            if random.random() < selection_prob
            else parent1.long_entry_conditions
        )

        child1_short_entry_conditions = (
            parent1.short_entry_conditions
            if random.random() < selection_prob
            else parent2.short_entry_conditions
        )
        child2_short_entry_conditions = (
            parent2.short_entry_conditions
            if random.random() < selection_prob
            else parent1.short_entry_conditions
        )

        child1_risk_management = (
            parent1.risk_management
            if random.random() < selection_prob
            else parent2.risk_management
        )
        child2_risk_management = (
            parent2.risk_management
            if random.random() < selection_prob
            else parent1.risk_management
        )

        child1_tpsl_gene = (
            parent1.tpsl_gene if random.random() < selection_prob else parent2.tpsl_gene
        )
        child2_tpsl_gene = (
            parent2.tpsl_gene if random.random() < selection_prob else parent1.tpsl_gene
        )

        child1_long_tpsl_gene = (
            parent1.long_tpsl_gene
            if random.random() < selection_prob
            else parent2.long_tpsl_gene
        )
        child2_long_tpsl_gene = (
            parent2.long_tpsl_gene
            if random.random() < selection_prob
            else parent1.long_tpsl_gene
        )

        child1_short_tpsl_gene = (
            parent1.short_tpsl_gene
            if random.random() < selection_prob
            else parent2.short_tpsl_gene
        )
        child2_short_tpsl_gene = (
            parent2.short_tpsl_gene
            if random.random() < selection_prob
            else parent1.short_tpsl_gene
        )

        child1_position_sizing_gene = (
            parent1.position_sizing_gene
            if random.random() < selection_prob
            else parent2.position_sizing_gene
        )
        child2_position_sizing_gene = (
            parent2.position_sizing_gene
            if random.random() < selection_prob
            else parent1.position_sizing_gene
        )

        # ツール遺伝子の交叉
        parent1_tool_genes = getattr(parent1, "tool_genes", []) or []
        parent2_tool_genes = getattr(parent2, "tool_genes", []) or []
        child1_tool_genes = (
            copy.deepcopy(parent1_tool_genes)
            if random.random() < selection_prob
            else copy.deepcopy(parent2_tool_genes)
        )
        child2_tool_genes = (
            copy.deepcopy(parent2_tool_genes)
            if random.random() < selection_prob
            else copy.deepcopy(parent1_tool_genes)
        )

        # メタデータの交叉（共通ユーティリティ使用）
        from ..utils.gene_utils import prepare_crossover_metadata

        child1_metadata, child2_metadata = prepare_crossover_metadata(parent1, parent2)

        # 子遺伝子の作成
        child1 = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=child1_indicators,
            long_entry_conditions=child1_long_entry_conditions,
            short_entry_conditions=child1_short_entry_conditions,
            risk_management=child1_risk_management,
            tpsl_gene=child1_tpsl_gene,
            long_tpsl_gene=child1_long_tpsl_gene,
            short_tpsl_gene=child1_short_tpsl_gene,
            position_sizing_gene=child1_position_sizing_gene,
            tool_genes=child1_tool_genes,
            metadata=child1_metadata,
        )

        child2 = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=child2_indicators,
            long_entry_conditions=child2_long_entry_conditions,
            short_entry_conditions=child2_short_entry_conditions,
            risk_management=child2_risk_management,
            tpsl_gene=child2_tpsl_gene,
            long_tpsl_gene=child2_long_tpsl_gene,
            short_tpsl_gene=child2_short_tpsl_gene,
            position_sizing_gene=child2_position_sizing_gene,
            tool_genes=child2_tool_genes,
            metadata=child2_metadata,
        )

        return child1, child2

    except Exception as e:
        logger.error(f"uniform crossoverエラー: {e}")
        # エラー時は親をそのまま返す
        return parent1, parent2

    except Exception as e:
        logger.error(f"uniform crossoverエラー: {e}")
        # エラー時は親をそのまま返す
        return parent1, parent2


def crossover_strategy_genes(
    parent1: Union[StrategyGene, list], parent2: Union[StrategyGene, list], config
) -> tuple[Union[StrategyGene, list], Union[StrategyGene, list]]:
    """
    戦略遺伝子の交叉

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な交叉を実行します。
    DEAPのIndividualオブジェクト（リスト）とStrategyGeneオブジェクトの両方に対応。

    Args:
        parent1: 親1の戦略遺伝子（StrategyGeneまたはIndividualオブジェクト）
        parent2: 親2の戦略遺伝子（StrategyGeneまたはIndividualオブジェクト）
        config: GAConfigオブジェクト

    Returns:
        交叉後の子1、子2の戦略遺伝子のタプル
    """
    try:
        # 入力の型を記録（戻り値の型を決定するため）
        parent1_is_individual = isinstance(parent1, list) or hasattr(parent1, "fitness")
        parent2_is_individual = isinstance(parent2, list) or hasattr(parent2, "fitness")

        # IndividualオブジェクトをStrategyGeneに変換
        strategy_parent1 = _convert_to_strategy_gene(parent1)
        strategy_parent2 = _convert_to_strategy_gene(parent2)

        # 純粋関数で交叉を実行
        result_child1, result_child2 = crossover_strategy_genes_pure(
            strategy_parent1, strategy_parent2, config
        )

        # 元の型に応じて適切な形式で返す
        if parent1_is_individual or parent2_is_individual:
            # DEAPのIndividualクラスを取得
            from deap import creator

            individual_class = getattr(creator, "Individual", None)

            child1_result = _convert_to_individual(result_child1, individual_class)
            child2_result = _convert_to_individual(result_child2, individual_class)
            return child1_result, child2_result
        else:
            return result_child1, result_child2

    except Exception as e:
        logger.error(f"戦略遺伝子交叉エラー: {e}")
        # エラー時は親をそのまま返す
        return parent1, parent2


def mutate_strategy_gene_pure(
    gene: StrategyGene, config, mutation_rate: float = 0.1
) -> StrategyGene:
    """
    戦略遺伝子の突然変異（純粋版）

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な突然変異を実行します。
    StrategyGeneオブジェクトのみを扱い、DEAPライブラリに依存しません。

    Args:
        gene: 突然変異対象の戦略遺伝子（StrategyGeneオブジェクト）
        config: GAConfigオブジェクト
        mutation_rate: 突然変異率

    Returns:
        突然変異後の戦略遺伝子
    """
    try:
        # 深いコピーを作成
        mutated = copy.deepcopy(gene)

        # 指標遺伝子の突然変異
        _mutate_indicators(mutated, gene, mutation_rate, config)

        # 条件の突然変異
        _mutate_conditions(mutated, mutation_rate, config)

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
                            * random.uniform(min_risk_multiplier, max_risk_multiplier),
                        ),
                    )
                else:
                    # その他の数値設定
                    mutated.risk_management[key] = value * random.uniform(
                        min_risk_multiplier, max_risk_multiplier
                    )

        # TP/SL遺伝子の突然変異
        tpsl_gene = mutated.tpsl_gene
        if tpsl_gene:
            if random.random() < mutation_rate:
                mutated.tpsl_gene = mutate_tpsl_gene(tpsl_gene, mutation_rate)
        else:
            # TP/SL遺伝子が存在しない場合、低確率で新規作成
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
        ps_gene = getattr(mutated, "position_sizing_gene", None)
        if ps_gene:
            if random.random() < mutation_rate:
                mutated.position_sizing_gene = mutate_position_sizing_gene(
                    ps_gene, mutation_rate
                )
        else:
            # ポジションサイジング遺伝子が存在しない場合、低確率で新規作成
            if (
                random.random()
                < mutation_rate
                * config.position_sizing_gene_creation_probability_multiplier
            ):
                from ..genes import (
                    create_random_position_sizing_gene,
                )

                mutated.position_sizing_gene = create_random_position_sizing_gene()

        # ツール遺伝子の突然変異
        tool_genes = getattr(mutated, "tool_genes", None)
        if tool_genes:
            from ..tools import tool_registry

            for tool_gene in tool_genes:
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

        # 新しいIDを生成（突然変異による変化を示す）
        mutated.id = str(uuid.uuid4())

        return mutated

    except Exception as e:
        logger.error(f"戦略遺伝子突然変異エラー: {e}")
        # エラー時は元の遺伝子をそのまま返す
        return gene


def create_deap_crossover_wrapper(individual_class=None, config=None):
    """
    DEAPクロスオーバーラッパーを作成

    Args:
        individual_class: DEAPのIndividualクラス。Noneの場合はcreatorから取得
        config: GAConfigオブジェクト (必須)

    Returns:
        DEAPツールボックスに登録可能なクロスオーバー関数
    """
    if config is None:
        raise ValueError("configオブジェクトは必須です")

    def crossover_wrapper(parent1, parent2):
        """
        DEAP用のクロスオーバーラッパー
        """
        try:
            # IndividualオブジェクトをStrategyGeneに変換
            strategy_parent1 = _convert_to_strategy_gene(parent1)
            strategy_parent2 = _convert_to_strategy_gene(parent2)

            # 純粋クロスオーバー関数で交叉を実行
            child1_strategy, child2_strategy = crossover_strategy_genes_pure(
                strategy_parent1, strategy_parent2, config
            )

            # StrategyGeneをIndividualオブジェクトに変換
            if individual_class is not None:
                _individual_class = individual_class
            else:
                from deap import creator

                _individual_class = getattr(creator, "Individual", None)

            child1_individual = _convert_to_individual(
                child1_strategy, _individual_class
            )
            child2_individual = _convert_to_individual(
                child2_strategy, _individual_class
            )

            return child1_individual, child2_individual

        except Exception as e:
            logger.error(f"DEAPクロスオーバーラッパーエラー: {e}")
            # エラー時は親をそのまま返す
            return parent1, parent2

    return crossover_wrapper


def create_deap_mutate_wrapper(individual_class=None, population=None, config=None):
    """
    DEAP突然変異ラッパーを作成

    Args:
        individual_class: DEAPのIndividualクラス。Noneの場合はcreatorから取得
        population: 個体集団（適応的突然変異用）
        config: GAConfigオブジェクト (必須)

    Returns:
        DEAPツールボックスに登録可能な突然変異関数
    """
    if config is None:
        raise ValueError("configオブジェクトは必須です")

    def mutate_wrapper(individual):
        """
        DEAP用の突然変異ラッパー
        """
        try:
            # IndividualオブジェクトをStrategyGeneに変換
            strategy_gene = _convert_to_strategy_gene(individual)

            # 適応的突然変異を使用
            if population is not None:
                mutated_strategy = adaptive_mutate_strategy_gene_pure(
                    population, strategy_gene, config
                )
            else:
                # populationがない場合は通常のmutate
                mutated_strategy = mutate_strategy_gene_pure(strategy_gene, config)

            # StrategyGeneをIndividualオブジェクトに変換
            if individual_class is not None:
                _individual_class = individual_class
            else:
                from deap import creator

                _individual_class = getattr(creator, "Individual", None)

            mutated_individual = _convert_to_individual(
                mutated_strategy, _individual_class
            )

            return (mutated_individual,)

        except Exception as e:
            logger.error(f"DEAP突然変異ラッパーエラー: {e}")
            # エラー時は元の個体をそのまま返す
            return (individual,)

    return mutate_wrapper


def mutate_strategy_gene(
    gene: Union[StrategyGene, list], config, mutation_rate: float = 0.1
) -> Union[StrategyGene, list]:
    """
    戦略遺伝子の突然変異

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な突然変異を実行します。
    DEAPのIndividualオブジェクト（リスト）とStrategyGeneオブジェクトの両方に対応。

    Args:
        gene: 突然変異対象の戦略遺伝子（StrategyGeneまたはIndividualオブジェクト）
        config: GAConfigオブジェクト
        mutation_rate: 突然変異率

    Returns:
        突然変異後の戦略遺伝子
    """
    try:
        # 入力の型を記録（戻り値の型を決定するため）
        gene_is_individual = isinstance(gene, list) or hasattr(gene, "fitness")

        # IndividualオブジェクトをStrategyGeneに変換
        strategy_gene = _convert_to_strategy_gene(gene)

        # 純粋関数で突然変異を実行
        result_mutated = mutate_strategy_gene_pure(strategy_gene, config, mutation_rate)

        # 元の型に応じて適切な形式で返す
        if gene_is_individual:
            # DEAPのIndividualクラスを取得
            from deap import creator

            individual_class = getattr(creator, "Individual", None)

            return _convert_to_individual(result_mutated, individual_class)
        else:
            return result_mutated

    except Exception as e:
        logger.error(f"戦略遺伝子突然変異エラー: {e}")
        # エラー時は元の遺伝子をそのまま返す
        return gene


def adaptive_mutate_strategy_gene_pure(
    population: list, gene: StrategyGene, config, base_mutation_rate: float = 0.1
) -> StrategyGene:
    """
    適応的戦略遺伝子突然変異（純粋版）

    populationのfitness varianceに基づいてmutation_rateを動的に調整し、
    mutate_strategy_gene_pureを実行します。

    Args:
        population: 個体集団（fitnessを持つIndividualオブジェクトのリスト）
        gene: 突然変異対象の戦略遺伝子（StrategyGeneオブジェクト）
        config: GAConfigオブジェクト
        base_mutation_rate: 基準突然変異率

    Returns:
        適応的突然変異後の戦略遺伝子
    """
    try:
        # populationからfitnessを抽出
        fitnesses = []
        for ind in population:
            if hasattr(ind, "fitness") and ind.fitness and ind.fitness.values:
                fitnesses.append(ind.fitness.values[0])  # 最初のfitness値を使用

        if not fitnesses:
            # fitnessがない場合は基準rateを使用
            adaptive_rate = base_mutation_rate
        else:
            # fitnessの分散を計算
            variance = np.var(fitnesses)

            # 分散に基づいてrateを調整
            # 分散が高い（多様性が高い）場合：rateを低く
            # 分散が低い（収束している）場合：rateを高く
            variance_threshold = config.adaptive_mutation_variance_threshold

            if variance > variance_threshold:
                # 多様性が高い：rateを減少
                adaptive_rate = (
                    base_mutation_rate
                    * config.adaptive_mutation_rate_decrease_multiplier
                )
            else:
                # 収束している：rateを増加
                adaptive_rate = (
                    base_mutation_rate
                    * config.adaptive_mutation_rate_increase_multiplier
                )

            # 0.01-1.0の範囲にクリップ
            adaptive_rate = max(0.01, min(1.0, adaptive_rate))

        # mutate_strategy_gene_pureを実行
        mutated = mutate_strategy_gene_pure(gene, config, mutation_rate=adaptive_rate)

        # metadataに適応的rateを追加
        mutated.metadata["adaptive_mutation_rate"] = adaptive_rate

        return mutated

    except Exception as e:
        logger.error(f"適応的戦略遺伝子突然変異エラー: {e}")
        # エラー時は元のrateでmutate
        return mutate_strategy_gene_pure(gene, config, base_mutation_rate)





