"""
遺伝的演算子

戦略遺伝子の交叉・突然変異ロジックを担当します。
"""

import copy
import logging
import random
import uuid
from typing import Union

import numpy as np

from ..models.strategy_models import (
    StrategyGene,
    crossover_tpsl_genes,
    crossover_position_sizing_genes,
    create_random_tpsl_gene,
    mutate_tpsl_gene,
    mutate_position_sizing_gene,
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


def _mutate_indicators(mutated, gene, mutation_rate):
    """
    指標の突然変異を処理

    Args:
        mutated: 突然変異対象のStrategyGene
        gene: 元のStrategyGene
        mutation_rate: 突然変異率
    """
    # 指標パラメータの突然変異
    for i, indicator in enumerate(mutated.indicators):
        if random.random() < mutation_rate:
            for param_name, param_value in indicator.parameters.items():
                if (
                    isinstance(param_value, (int, float))
                    and random.random() < mutation_rate
                ):
                    if param_name == "period":
                        mutated.indicators[i].parameters[param_name] = max(
                            1, min(200, int(param_value * random.uniform(0.8, 1.2)))
                        )
                    else:
                        mutated.indicators[i].parameters[param_name] = (
                            param_value * random.uniform(0.8, 1.2)
                        )

    # 指標の追加・削除
    if random.random() < mutation_rate * 0.3:
        max_indicators = getattr(gene, "MAX_INDICATORS", 5)

        if len(mutated.indicators) < max_indicators and random.random() < 0.5:
            # 新しい指標を追加
            from ..generators.random_gene_generator import RandomGeneGenerator
            from ..config import GAConfig

            generator = RandomGeneGenerator(GAConfig())
            new_indicators = generator.indicator_generator.generate_random_indicators()
            if new_indicators:
                mutated.indicators.append(random.choice(new_indicators))

        elif len(mutated.indicators) > 1 and random.random() < 0.5:
            # 指標を削除
            mutated.indicators.pop(random.randint(0, len(mutated.indicators) - 1))


def _mutate_conditions(mutated, mutation_rate):
    """
    条件の突然変異を処理

    Args:
        mutated: 突然変異対象のStrategyGene
        mutation_rate: 突然変異率
    """
    if random.random() < mutation_rate * 0.5:
        # エントリー条件の変更
        if mutated.entry_conditions and random.random() < 0.5:
            condition_idx = random.randint(0, len(mutated.entry_conditions) - 1)
            condition = mutated.entry_conditions[condition_idx]

            operators = [">", "<", ">=", "<=", "=="]
            condition.operator = random.choice(operators)

    if random.random() < mutation_rate * 0.5:
        # エグジット条件の変更
        if mutated.exit_conditions and random.random() < 0.5:
            condition_idx = random.randint(0, len(mutated.exit_conditions) - 1)
            condition = mutated.exit_conditions[condition_idx]

            operators = [">", "<", ">=", "<=", "=="]
            condition.operator = random.choice(operators)


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


def crossover_strategy_genes_pure(
    parent1: StrategyGene, parent2: StrategyGene, crossover_type: str = "uniform"
) -> tuple[StrategyGene, StrategyGene]:
    """
    戦略遺伝子の交叉（純粋版）

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な交叉を実行します。
    StrategyGeneオブジェクトのみを扱い、DEAPライブラリに依存しません。

    Args:
        parent1: 親1の戦略遺伝子（StrategyGeneオブジェクト）
        parent2: 親2の戦略遺伝子（StrategyGeneオブジェクト）
        crossover_type: 交叉タイプ ("single_point" または "uniform")

    Returns:
        交叉後の子1、子2の戦略遺伝子のタプル
    """
    try:
        if crossover_type == "uniform":
            return uniform_crossover(parent1, parent2)
        else:
            # single_point crossover (existing logic)
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
            max_indicators = getattr(parent1, "MAX_INDICATORS", 5)
            child1_indicators = child1_indicators[:max_indicators]
            child2_indicators = child2_indicators[:max_indicators]

            # 条件の交叉（ランダム選択）
            if random.random() < 0.5:
                child1_entry = parent1.entry_conditions.copy()
                child2_entry = parent2.entry_conditions.copy()
            else:
                child1_entry = parent2.entry_conditions.copy()
                child2_entry = parent1.entry_conditions.copy()

            if random.random() < 0.5:
                child1_exit = parent1.exit_conditions.copy()
                child2_exit = parent2.exit_conditions.copy()
            else:
                child1_exit = parent2.exit_conditions.copy()
                child2_exit = parent1.exit_conditions.copy()

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

            return child1_strategy, child2_strategy

    except Exception as e:
        logger.error(f"戦略遺伝子交叉エラー: {e}")
        # エラー時は親をそのまま返す
        return parent1, parent2


def uniform_crossover(
    parent1: StrategyGene, parent2: StrategyGene
) -> tuple[StrategyGene, StrategyGene]:
    """
    ユニフォーム交叉

    StrategyGeneの各フィールドについて、各遺伝子位置でランダムに親を選択します。
    多様性を高めるために使用されます。

    Args:
        parent1: 親1の戦略遺伝子（StrategyGeneオブジェクト）
        parent2: 親2の戦略遺伝子（StrategyGeneオブジェクト）

    Returns:
        交叉後の子1、子2の戦略遺伝子のタプル
    """
    try:
        # 各フィールドに対してランダム選択
        child1_indicators = (
            parent1.indicators if random.random() < 0.5 else parent2.indicators
        )
        child2_indicators = (
            parent2.indicators if random.random() < 0.5 else parent1.indicators
        )

        child1_entry_conditions = (
            parent1.entry_conditions
            if random.random() < 0.5
            else parent2.entry_conditions
        )
        child2_entry_conditions = (
            parent2.entry_conditions
            if random.random() < 0.5
            else parent1.entry_conditions
        )

        child1_exit_conditions = (
            parent1.exit_conditions
            if random.random() < 0.5
            else parent2.exit_conditions
        )
        child2_exit_conditions = (
            parent2.exit_conditions
            if random.random() < 0.5
            else parent1.exit_conditions
        )

        child1_long_entry_conditions = (
            parent1.long_entry_conditions
            if random.random() < 0.5
            else parent2.long_entry_conditions
        )
        child2_long_entry_conditions = (
            parent2.long_entry_conditions
            if random.random() < 0.5
            else parent1.long_entry_conditions
        )

        child1_short_entry_conditions = (
            parent1.short_entry_conditions
            if random.random() < 0.5
            else parent2.short_entry_conditions
        )
        child2_short_entry_conditions = (
            parent2.short_entry_conditions
            if random.random() < 0.5
            else parent1.short_entry_conditions
        )

        child1_risk_management = (
            parent1.risk_management
            if random.random() < 0.5
            else parent2.risk_management
        )
        child2_risk_management = (
            parent2.risk_management
            if random.random() < 0.5
            else parent1.risk_management
        )

        child1_tpsl_gene = (
            parent1.tpsl_gene if random.random() < 0.5 else parent2.tpsl_gene
        )
        child2_tpsl_gene = (
            parent2.tpsl_gene if random.random() < 0.5 else parent1.tpsl_gene
        )

        child1_position_sizing_gene = (
            parent1.position_sizing_gene
            if random.random() < 0.5
            else parent2.position_sizing_gene
        )
        child2_position_sizing_gene = (
            parent2.position_sizing_gene
            if random.random() < 0.5
            else parent1.position_sizing_gene
        )

        # メタデータの交叉（共通ユーティリティ使用）
        from ..utils.gene_utils import prepare_crossover_metadata

        child1_metadata, child2_metadata = prepare_crossover_metadata(parent1, parent2)

        # 子遺伝子の作成
        child1 = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=child1_indicators,
            entry_conditions=child1_entry_conditions,
            exit_conditions=child1_exit_conditions,
            long_entry_conditions=child1_long_entry_conditions,
            short_entry_conditions=child1_short_entry_conditions,
            risk_management=child1_risk_management,
            tpsl_gene=child1_tpsl_gene,
            position_sizing_gene=child1_position_sizing_gene,
            metadata=child1_metadata,
        )

        child2 = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=child2_indicators,
            entry_conditions=child2_entry_conditions,
            exit_conditions=child2_exit_conditions,
            long_entry_conditions=child2_long_entry_conditions,
            short_entry_conditions=child2_short_entry_conditions,
            risk_management=child2_risk_management,
            tpsl_gene=child2_tpsl_gene,
            position_sizing_gene=child2_position_sizing_gene,
            metadata=child2_metadata,
        )

        return child1, child2

    except Exception as e:
        logger.error(f"uniform crossoverエラー: {e}")
        # エラー時は親をそのまま返す
        return parent1, parent2


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

        # 純粋関数で交叉を実行
        result_child1, result_child2 = crossover_strategy_genes_pure(
            strategy_parent1, strategy_parent2
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
    gene: StrategyGene, mutation_rate: float = 0.1
) -> StrategyGene:
    """
    戦略遺伝子の突然変異（純粋版）

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な突然変異を実行します。
    StrategyGeneオブジェクトのみを扱い、DEAPライブラリに依存しません。

    Args:
        gene: 突然変異対象の戦略遺伝子（StrategyGeneオブジェクト）
        mutation_rate: 突然変異率

    Returns:
        突然変異後の戦略遺伝子
    """
    try:
        # 深いコピーを作成
        mutated = copy.deepcopy(gene)

        # 指標遺伝子の突然変異
        _mutate_indicators(mutated, gene, mutation_rate)

        # 条件の突然変異
        _mutate_conditions(mutated, mutation_rate)

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
                mutated.tpsl_gene = mutate_tpsl_gene(tpsl_gene, mutation_rate)
        else:
            # TP/SL遺伝子が存在しない場合、低確率で新規作成
            if random.random() < mutation_rate * 0.2:
                mutated.tpsl_gene = create_random_tpsl_gene()

        # ポジションサイジング遺伝子の突然変異
        ps_gene = getattr(mutated, "position_sizing_gene", None)
        if ps_gene:
            if random.random() < mutation_rate:
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

        # 新しいIDを生成（突然変異による変化を示す）
        mutated.id = str(uuid.uuid4())

        return mutated

    except Exception as e:
        logger.error(f"戦略遺伝子突然変異エラー: {e}")
        # エラー時は元の遺伝子をそのまま返す
        return gene


def create_deap_crossover_wrapper(individual_class=None):
    """
    DEAPクロスオーバーラッパーを作成

    Args:
        individual_class: DEAPのIndividualクラス。Noneの場合はcreatorから取得

    Returns:
        DEAPツールボックスに登録可能なクロスオーバー関数
    """

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
                strategy_parent1, strategy_parent2
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


def create_deap_mutate_wrapper(individual_class=None, population=None):
    """
    DEAP突然変異ラッパーを作成

    Args:
        individual_class: DEAPのIndividualクラス。Noneの場合はcreatorから取得
        population: 個体集団（適応的突然変異用）

    Returns:
        DEAPツールボックスに登録可能な突然変異関数
    """

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
                    population, strategy_gene
                )
            else:
                # populationがない場合は通常のmutate
                mutated_strategy = mutate_strategy_gene_pure(strategy_gene)

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

        # 純粋関数で突然変異を実行
        result_mutated = mutate_strategy_gene_pure(strategy_gene, mutation_rate)

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
    population: list, gene: StrategyGene, base_mutation_rate: float = 0.1
) -> StrategyGene:
    """
    適応的戦略遺伝子突然変異（純粋版）

    populationのfitness varianceに基づいてmutation_rateを動的に調整し、
    mutate_strategy_gene_pureを実行します。

    Args:
        population: 個体集団（fitnessを持つIndividualオブジェクトのリスト）
        gene: 突然変異対象の戦略遺伝子（StrategyGeneオブジェクト）
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
            variance_threshold = 0.1  # 適当な閾値

            if variance > variance_threshold:
                # 多様性が高い：rateを半分に
                adaptive_rate = base_mutation_rate * 0.5
            else:
                # 収束している：rateを2倍に
                adaptive_rate = base_mutation_rate * 2.0

            # 0.01-1.0の範囲にクリップ
            adaptive_rate = max(0.01, min(1.0, adaptive_rate))

        # mutate_strategy_gene_pureを実行
        mutated = mutate_strategy_gene_pure(gene, mutation_rate=adaptive_rate)

        # metadataに適応的rateを追加
        mutated.metadata["adaptive_mutation_rate"] = adaptive_rate

        return mutated

    except Exception as e:
        logger.error(f"適応的戦略遺伝子突然変異エラー: {e}")
        # エラー時は元のrateでmutate
        return mutate_strategy_gene_pure(gene, base_mutation_rate)
