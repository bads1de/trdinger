"""
遺伝的アルゴリズムユーティリティ関数モジュール

GAエンジンで使用される共通のヘルパー関数を提供します。
"""

import logging
from typing import Any, Dict

from app.services.auto_strategy.genes.genetic_utils import GeneticUtils

logger = logging.getLogger(__name__)


def crossover_strategy_genes(parent1, parent2, config):
    """
    戦略遺伝子の交叉ラッパー

    親個体の交叉メソッドを呼び出して、子個体を生成します。

    Args:
        parent1: 親個体1
        parent2: 親個体2
        config: GA設定オブジェクト

    Returns:
        交叉後の個体（タプル形式、(child1, child2)）

    Note:
        StrategyGeneクラスのcrossoverクラスメソッドを使用します。
    """
    return type(parent1).crossover(parent1, parent2, config)


def mutate_strategy_gene(gene, config, mutation_rate=0.1):
    """
    戦略遺伝子の突然変異ラッパー

    遺伝子の突然変異メソッドを呼び出して、変異後の遺伝子を生成します。

    Args:
        gene: 突然変異対象の遺伝子
        config: GA設定オブジェクト
        mutation_rate: 突然変異率（デフォルト: 0.1）

    Returns:
        突然変異後の遺伝子

    Note:
        StrategyGeneクラスのmutateメソッドを使用します。
    """
    return gene.mutate(config, mutation_rate)


def _gene_kwargs(gene) -> Dict[str, Any]:
    """
    StrategyGene系オブジェクトをコンストラクタ用のkwargsに変換

    遺伝子オブジェクトからパラメータを抽出して、
    コンストラクタ用のkwargs辞書に変換します。

    Args:
        gene: 遺伝子オブジェクト

    Returns:
        Dict[str, Any]: コンストラクタ用パラメータ辞書

    Note:
        GeneticUtils.extract_gene_paramsを使用します。
    """
    return GeneticUtils.extract_gene_params(gene)


def _invalidate_individual_cache(individual) -> None:
    """
    評価済み個体のキャッシュを無効化

    個体のキャッシュ（fitness.values、feature_vector）を削除して、
    再評価を強制します。

    Args:
        individual: 個体オブジェクト

    Note:
        fitness.valuesと_feature_vector属性を削除します。
    """
    if hasattr(individual, "fitness") and hasattr(individual.fitness, "values"):
        del individual.fitness.values
    if hasattr(individual, "_feature_vector"):
        del individual._feature_vector


def _set_fitness_values(individuals, fitnesses) -> None:
    """
    個体列に fitness.values をまとめて設定

    個体リストとフィットネスリストを対応させて、
    各個体のfitness.valuesを設定します。

    Args:
        individuals: 個体リスト
        fitnesses: フィットネス値リスト

    Note:
        リストの長さが一致している必要があります。
    """
    for ind, fit in zip(individuals, fitnesses):
        ind.fitness.values = fit


def create_deap_mutate_wrapper(individual_class, population, config):
    """DEAP用の突然変異ラッパー関数を作成する。

    適応的突然変異（Adaptive Mutation）をサポートするためのクロージャを返します。
    集団の状態に基づいて突然変異率を調整します。

    Args:
        individual_class: 生成する個体クラス（StrategyGeneを継承）。
        population: 現在の集団（適応的突然変異用）。個体のフィットネス分散を計算するために使用。
        config: GA設定オブジェクト。mutation_rate、adaptive_mutation関連の設定を含む。

    Returns:
        Callable: DEAPに登録可能な突然変異ラッパー関数。
            シグネチャ: (individual) -> tuple[StrategyGene]
            入力: 突然変異を適用する個体。
            出力: 突然変異後の個体を要素に持つタプル（DEAP要件）。

    Note:
        - 集団がある場合は適応的突然変異を使用
        - 集団がない場合は通常の突然変異を使用
        - DEAPの要件に合わせてタプルで返します
    """

    def mutate_wrapper(individual):
        """適応的突然変異を実行するラッパー。

        集団の状態に基づいて突然変異率を調整し、突然変異を実行する。
        DEAPの要件に合わせて、突然変異後の個体をタプルでラップして返す。
        """
        # 適応的突然変異を使用
        if population is not None:
            # individual自体がStrategyGeneのインスタンス
            mutated_strategy = individual.adaptive_mutate(
                population, config, base_mutation_rate=config.mutation_rate
            )
        else:
            mutated_strategy = individual.mutate(
                config, mutation_rate=config.mutation_rate
            )

        # StrategyGeneをIndividualに変換
        # StrategyGeneを継承しているため、フィールドを展開して初期化
        return (individual_class(**_gene_kwargs(mutated_strategy)),)

    return mutate_wrapper
