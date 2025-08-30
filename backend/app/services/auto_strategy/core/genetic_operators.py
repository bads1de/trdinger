"""
遺伝的演算子

戦略遺伝子の交叉・突然変異ロジックを担当します。
"""

import copy
import logging
import random
import uuid
from typing import Union, overload

from ..models.strategy_models import StrategyGene

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
        from ..serializers.gene_serialization import GeneSerializer

        gene_serializer = GeneSerializer()
        encoded_gene = gene_serializer.encode_strategy_gene_to_list(strategy_gene)
        return individual_class(encoded_gene)
    except Exception as e:
        logger.error(f"StrategyGene→Individual変換エラー: {e}")
        raise


@overload
def crossover_strategy_genes(
    parent1: StrategyGene, parent2: StrategyGene
) -> tuple[StrategyGene, StrategyGene]: ...


@overload
def crossover_strategy_genes(parent1: list, parent2: list) -> tuple[list, list]: ...


def crossover_strategy_genes(
    parent1: Union[StrategyGene, list], parent2: Union[StrategyGene, list]
) -> tuple[Union[StrategyGene, list], Union[StrategyGene, list]]:
    """
    戦略遺伝子の交叉

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な交叉を実行します。
    DEAPのIndividualオブジェクト（リスト）とStrategyGeneオブジェクトの両方に対応。

    Args:
        parent1: 親1の戦略遺伝子（StrategyGeneまたはIndividualオブジェクト）
        parent2: 親2の戦略遺伝子（StrategyGeneまたはIndividualオブジェクト）

    Returns:
        交叉後の子1、子2の戦略遺伝子のタプル
    """
    try:
        # 入力の型を記録（戻り値の型を決定するため）
        parent1_is_individual = isinstance(parent1, list)
        parent2_is_individual = isinstance(parent2, list)

        # IndividualオブジェクトをStrategyGeneに変換
        strategy_parent1 = _convert_to_strategy_gene(parent1)
        strategy_parent2 = _convert_to_strategy_gene(parent2)
        # 指標遺伝子の交叉（単純な一点交叉）
        min_indicators = min(
            len(strategy_parent1.indicators), len(strategy_parent2.indicators)
        )
        if min_indicators <= 1:
            # 指標数が1以下の場合は交叉点を0に設定（全体を交換）
            crossover_point = 0
        else:
            crossover_point = random.randint(1, min_indicators)

        child1_indicators = (
            strategy_parent1.indicators[:crossover_point]
            + strategy_parent2.indicators[crossover_point:]
        )
        child2_indicators = (
            strategy_parent2.indicators[:crossover_point]
            + strategy_parent1.indicators[crossover_point:]
        )

        # 最大指標数制限
        max_indicators = getattr(strategy_parent1, "MAX_INDICATORS", 5)
        child1_indicators = child1_indicators[:max_indicators]
        child2_indicators = child2_indicators[:max_indicators]

        # 条件の交叉（ランダム選択）
        if random.random() < 0.5:
            child1_entry = strategy_parent1.entry_conditions.copy()
            child2_entry = strategy_parent2.entry_conditions.copy()
        else:
            child1_entry = strategy_parent2.entry_conditions.copy()
            child2_entry = strategy_parent1.entry_conditions.copy()

        if random.random() < 0.5:
            child1_exit = strategy_parent1.exit_conditions.copy()
            child2_exit = strategy_parent2.exit_conditions.copy()
        else:
            child1_exit = strategy_parent2.exit_conditions.copy()
            child2_exit = strategy_parent1.exit_conditions.copy()

        # リスク管理設定の交叉（平均値）
        child1_risk = {}
        child2_risk = {}

        all_keys = set(strategy_parent1.risk_management.keys()) | set(
            strategy_parent2.risk_management.keys()
        )
        for key in all_keys:
            val1 = strategy_parent1.risk_management.get(key, 0)
            val2 = strategy_parent2.risk_management.get(key, 0)

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                child1_risk[key] = (val1 + val2) / 2
                child2_risk[key] = (val1 + val2) / 2
            else:
                child1_risk[key] = val1 if random.random() < 0.5 else val2
                child2_risk[key] = val2 if random.random() < 0.5 else val1

        # TP/SL遺伝子の交叉
        child1_tpsl = None
        child2_tpsl = None

        if strategy_parent1.tpsl_gene and strategy_parent2.tpsl_gene:
            from ..models.strategy_models import crossover_tpsl_genes

            child1_tpsl, child2_tpsl = crossover_tpsl_genes(
                strategy_parent1.tpsl_gene, strategy_parent2.tpsl_gene
            )
        elif strategy_parent1.tpsl_gene:
            child1_tpsl = strategy_parent1.tpsl_gene
            child2_tpsl = strategy_parent1.tpsl_gene  # コピー
        elif strategy_parent2.tpsl_gene:
            child1_tpsl = strategy_parent2.tpsl_gene
            child2_tpsl = strategy_parent2.tpsl_gene  # コピー

        # ポジションサイジング遺伝子の交叉
        child1_position_sizing = None
        child2_position_sizing = None

        ps_gene1 = getattr(strategy_parent1, "position_sizing_gene", None)
        ps_gene2 = getattr(strategy_parent2, "position_sizing_gene", None)

        if ps_gene1 and ps_gene2:
            from ..models.strategy_models import crossover_position_sizing_genes

            child1_position_sizing, child2_position_sizing = (
                crossover_position_sizing_genes(ps_gene1, ps_gene2)
            )
        elif ps_gene1:
            child1_position_sizing = ps_gene1
            child2_position_sizing = copy.deepcopy(ps_gene1)
        elif ps_gene2:
            child1_position_sizing = ps_gene2
            child2_position_sizing = copy.deepcopy(ps_gene2)

        # メタデータの交叉（共通ユーティリティ使用）
        from ..utils.common_utils import prepare_crossover_metadata

        child1_metadata, child2_metadata = prepare_crossover_metadata(
            strategy_parent1, strategy_parent2
        )

        # ロング・ショート条件の交叉
        if random.random() < 0.5:
            child1_long_entry = strategy_parent1.long_entry_conditions.copy()
            child2_long_entry = strategy_parent2.long_entry_conditions.copy()
        else:
            child1_long_entry = strategy_parent2.long_entry_conditions.copy()
            child2_long_entry = strategy_parent1.long_entry_conditions.copy()

        if random.random() < 0.5:
            child1_short_entry = strategy_parent1.short_entry_conditions.copy()
            child2_short_entry = strategy_parent2.short_entry_conditions.copy()
        else:
            child1_short_entry = strategy_parent2.short_entry_conditions.copy()
            child2_short_entry = strategy_parent1.short_entry_conditions.copy()

        # 子遺伝子の作成
        child1_strategy = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=child1_indicators,
            entry_conditions=child1_entry,
            exit_conditions=child1_exit,
            long_entry_conditions=child1_long_entry,
            short_entry_conditions=child1_short_entry,
            risk_management=child1_risk,
            tpsl_gene=child1_tpsl,
            position_sizing_gene=child1_position_sizing,
            metadata=child1_metadata,
        )

        child2_strategy = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=child2_indicators,
            entry_conditions=child2_entry,
            exit_conditions=child2_exit,
            long_entry_conditions=child2_long_entry,
            short_entry_conditions=child2_short_entry,
            risk_management=child2_risk,
            tpsl_gene=child2_tpsl,
            position_sizing_gene=child2_position_sizing,
            metadata=child2_metadata,
        )

        # 元の型に応じて適切な形式で返す
        if parent1_is_individual or parent2_is_individual:
            # DEAPのIndividualクラスを取得
            from deap import creator

            individual_class = getattr(creator, "Individual", None)

            child1_result = _convert_to_individual(child1_strategy, individual_class)
            child2_result = _convert_to_individual(child2_strategy, individual_class)
            return child1_result, child2_result
        else:
            return child1_strategy, child2_strategy

    except Exception as e:
        logger.error(f"戦略遺伝子交叉エラー: {e}")
        # エラー時は親をそのまま返す
        return parent1, parent2


@overload
def mutate_strategy_gene(
    gene: StrategyGene, mutation_rate: float = 0.1
) -> StrategyGene: ...


@overload
def mutate_strategy_gene(gene: list, mutation_rate: float = 0.1) -> list: ...


def mutate_strategy_gene(
    gene: Union[StrategyGene, list], mutation_rate: float = 0.1
) -> Union[StrategyGene, list]:
    """
    戦略遺伝子の突然変異

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な突然変異を実行します。
    DEAPのIndividualオブジェクト（リスト）とStrategyGeneオブジェクトの両方に対応。

    Args:
        gene: 突然変異対象の戦略遺伝子（StrategyGeneまたはIndividualオブジェクト）
        mutation_rate: 突然変異率

    Returns:
        突然変異後の戦略遺伝子
    """
    try:
        # 入力の型を記録（戻り値の型を決定するため）
        gene_is_individual = isinstance(gene, list)

        # IndividualオブジェクトをStrategyGeneに変換
        strategy_gene = _convert_to_strategy_gene(gene)

        # 深いコピーを作成
        mutated = copy.deepcopy(strategy_gene)

        # 指標遺伝子の突然変異
        for i, indicator in enumerate(mutated.indicators):
            if random.random() < mutation_rate:
                # パラメータの突然変異
                for param_name, param_value in indicator.parameters.items():
                    if (
                        isinstance(param_value, (int, float))
                        and random.random() < mutation_rate
                    ):
                        if param_name == "period":
                            # 期間パラメータの場合
                            mutated.indicators[i].parameters[param_name] = max(
                                1, min(200, int(param_value * random.uniform(0.8, 1.2)))
                            )
                        else:
                            # その他の数値パラメータ
                            mutated.indicators[i].parameters[param_name] = (
                                param_value * random.uniform(0.8, 1.2)
                            )

        # 指標の追加・削除（低確率）
        if random.random() < mutation_rate * 0.3:
            max_indicators = getattr(gene, "MAX_INDICATORS", 5)

            if len(mutated.indicators) < max_indicators and random.random() < 0.5:
                # 新しい指標を追加
                from ..generators.random_gene_generator import RandomGeneGenerator
                from ..config.auto_strategy_config import GAConfig

                generator = RandomGeneGenerator(GAConfig())
                new_indicators = generator._generate_random_indicators()
                if new_indicators:
                    mutated.indicators.append(random.choice(new_indicators))

            elif len(mutated.indicators) > 1 and random.random() < 0.5:
                # 指標を削除
                mutated.indicators.pop(random.randint(0, len(mutated.indicators) - 1))

        # 条件の突然変異（低確率）
        if random.random() < mutation_rate * 0.5:
            # エントリー条件の変更
            if mutated.entry_conditions and random.random() < 0.5:
                condition_idx = random.randint(0, len(mutated.entry_conditions) - 1)
                condition = mutated.entry_conditions[condition_idx]

                # オペレーターの変更
                operators = [">", "<", ">=", "<=", "=="]
                condition.operator = random.choice(operators)

        if random.random() < mutation_rate * 0.5:
            # エグジット条件の変更
            if mutated.exit_conditions and random.random() < 0.5:
                condition_idx = random.randint(0, len(mutated.exit_conditions) - 1)
                condition = mutated.exit_conditions[condition_idx]

                # オペレーターの変更
                operators = [">", "<", ">=", "<=", "=="]
                condition.operator = random.choice(operators)

        # リスク管理設定の突然変異
        for key, value in mutated.risk_management.items():
            if isinstance(value, (int, float)) and random.random() < mutation_rate:
                if key == "position_size":
                    # ポジションサイズの場合
                    mutated.risk_management[key] = max(
                        0.01, min(1.0, value * random.uniform(0.8, 1.2))
                    )
                else:
                    # その他の数値設定
                    mutated.risk_management[key] = value * random.uniform(0.8, 1.2)

        # TP/SL遺伝子の突然変異
        tpsl_gene = mutated.tpsl_gene
        if tpsl_gene:
            if random.random() < mutation_rate:
                from ..models.strategy_models import mutate_tpsl_gene

                mutated.tpsl_gene = mutate_tpsl_gene(tpsl_gene, mutation_rate)
        else:
            # TP/SL遺伝子が存在しない場合、低確率で新規作成
            if random.random() < mutation_rate * 0.2:
                from ..models.strategy_models import create_random_tpsl_gene

                mutated.tpsl_gene = create_random_tpsl_gene()

        # ポジションサイジング遺伝子の突然変異
        ps_gene = getattr(mutated, "position_sizing_gene", None)
        if ps_gene:
            if random.random() < mutation_rate:
                from ..models.strategy_models import mutate_position_sizing_gene

                mutated.position_sizing_gene = mutate_position_sizing_gene(
                    ps_gene, mutation_rate
                )
        else:
            # ポジションサイジング遺伝子が存在しない場合、低確率で新規作成
            if random.random() < mutation_rate * 0.2:
                from ..models.strategy_models import (
                    create_random_position_sizing_gene,
                )

                mutated.position_sizing_gene = create_random_position_sizing_gene()

        # メタデータの更新
        mutated.metadata["mutated"] = True
        mutated.metadata["mutation_rate"] = mutation_rate

        # 元の型に応じて適切な形式で返す
        if gene_is_individual:
            # DEAPのIndividualクラスを取得
            from deap import creator

            individual_class = getattr(creator, "Individual", None)

            return _convert_to_individual(mutated, individual_class)
        else:
            return mutated

    except Exception as e:
        logger.error(f"戦略遺伝子突然変異エラー: {e}")
        # エラー時は元の遺伝子をそのまま返す
        return gene
