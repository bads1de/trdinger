"""
遺伝的アルゴリズムユーティリティ関数モジュール

GAエンジンで使用される共通のヘルパー関数を提供します。
"""

import logging
from typing import Any, Dict

from ..genes.genetic_utils import GeneticUtils

logger = logging.getLogger(__name__)


def crossover_strategy_genes(parent1, parent2, config):
    """
    戦略遺伝子の交叉ラッパー

    Args:
        parent1: 親個体1
        parent2: 親個体2
        config: GA設定

    Returns:
        交叉後の個体（タプル形式、(child1, child2)）
    """
    return type(parent1).crossover(parent1, parent2, config)


def mutate_strategy_gene(gene, config, mutation_rate=0.1):
    """
    戦略遺伝子の突然変異ラッパー

    Args:
        gene: 突然変異対象の遺伝子
        config: GA設定
        mutation_rate: 突然変異率

    Returns:
        突然変異後の遺伝子
    """
    return gene.mutate(config, mutation_rate)


def _gene_kwargs(gene) -> Dict[str, Any]:
    """StrategyGene系オブジェクトをコンストラクタ用のkwargsに変換"""
    return GeneticUtils.extract_gene_params(gene)


def _invalidate_individual_cache(individual) -> None:
    """評価済み個体のキャッシュを無効化"""
    if hasattr(individual, "fitness") and hasattr(individual.fitness, "values"):
        del individual.fitness.values
    if hasattr(individual, "_feature_vector"):
        del individual._feature_vector


def _set_fitness_values(individuals, fitnesses) -> None:
    """個体列に fitness.values をまとめて設定"""
    for ind, fit in zip(individuals, fitnesses):
        ind.fitness.values = fit


def create_deap_mutate_wrapper(individual_class, population, config):
    """
    DEAP用の突然変異ラッパー関数を作成します。

    適応的突然変異（Adaptive Mutation）をサポートするためのクロージャを返します。

    Args:
        individual_class: 生成する個体クラス
        population: 現在の集団（適応的突然変異用）
        config: GA設定

    Returns:
        DEAPに登録可能な突然変異ラッパー関数
    """

    def mutate_wrapper(individual):
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
