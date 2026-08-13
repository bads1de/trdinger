"""
遺伝的アルゴリズムユーティリティ関数モジュール

GAエンジンで使用される共通のヘルパー関数を提供します。
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.services.auto_strategy.genes.genetic_utils import GeneticUtils

if TYPE_CHECKING:
    from app.services.auto_strategy.config.ga_config import GAConfig
    from app.services.auto_strategy.genes.strategy import StrategyGene

logger = logging.getLogger(__name__)


def crossover_strategy_genes(
    parent1: StrategyGene, parent2: StrategyGene, config: GAConfig
) -> tuple[StrategyGene, StrategyGene]:
    """戦略遺伝子の交叉を実行する。

    親個体のクラスメソッド ``crossover`` を呼び出して子個体を生成します。

    Args:
        parent1: 親個体1
        parent2: 親個体2
        config: GA設定オブジェクト

    Returns:
        交叉後の個体のタプル ``(child1, child2)``
    """
    return type(parent1).crossover(parent1, parent2, config)


def _invalidate_individual_cache(  # pyright: ignore[reportUnusedFunction] - evolution_runner やテストから import して使用
    individual: Any,
) -> None:
    """評価済み個体のキャッシュを無効化する。

    ``fitness.values`` と ``_feature_vector`` を削除し、次回アクセス時に
    再評価・再計算を強制します。併せて個体に紐づく ``_cache`` もクリアします。

    Args:
        individual: 個体オブジェクト
    """
    with contextlib.suppress(AttributeError):
        del individual.fitness.values
    with contextlib.suppress(AttributeError):
        del individual._feature_vector
    with contextlib.suppress(AttributeError):
        individual._cache = {}


def _set_fitness_values(  # pyright: ignore[reportUnusedFunction] - evolution_runner やテストから import して使用
    individuals: list[Any], fitnesses: list[tuple[float, ...]]
) -> None:
    """個体群にフィットネス値をまとめて設定する。

    Args:
        individuals: 個体リスト
        fitnesses: フィットネス値リスト

    Raises:
        ValueError: リストの長さが異なる場合
    """
    if len(individuals) != len(fitnesses):
        raise ValueError(
            f"個体数({len(individuals)})とフィットネス数({len(fitnesses)})が一致しません"
        )
    for ind, fit in zip(individuals, fitnesses, strict=False):
        ind.fitness.values = fit


def create_deap_mutate_wrapper(
    individual_class: type,
    population: list[StrategyGene] | None,
    config: GAConfig,
) -> Callable[[StrategyGene], tuple[StrategyGene]]:
    """DEAP用の突然変異ラッパー関数を作成する。

    適応的突然変異（Adaptive Mutation）をサポートするクロージャを返します。
    集団の状態に基づいて突然変異率を調整します。

    Args:
        individual_class: 生成する個体クラス（StrategyGeneを継承）。
        population: 現在の集団（適応的突然変異用）。Noneの場合は通常の突然変異を使用。
        config: GA設定オブジェクト。mutation_rate、adaptive_mutation関連の設定を含む。

    Returns:
        DEAPに登録可能な突然変異ラッパー関数。
        シグネチャ: ``(individual) -> tuple[StrategyGene]``
    """

    def mutate_wrapper(individual: StrategyGene) -> tuple[StrategyGene]:
        if population is not None:
            mutated_strategy = individual.adaptive_mutate(
                population, config, base_mutation_rate=config.mutation_rate
            )
        else:
            mutated_strategy = individual.mutate(
                config, mutation_rate=config.mutation_rate
            )

        return (individual_class(**GeneticUtils.extract_gene_params(mutated_strategy)),)

    return mutate_wrapper
