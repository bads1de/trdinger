"""
フィットネス共有のシルエットベース調整ユーティリティ
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.serializers.serialization import GeneSerializer

from .fitness_validation import (
    compute_worst_selection_values,
    has_valid_fitness,
    resolve_minimize_flags,
    shrink_advantage_toward_worst,
)

logger = logging.getLogger(__name__)

# 定数
MIN_VECTORS_FOR_CLUSTERING = 3
MAX_CLUSTERS = 3
RANDOM_STATE = 42
SILHOUETTE_OFFSET = 1.0
SILHOUETTE_DIVISOR = 2.0
# 適応度調整倍率の下限。クラスタ中心の個体でも生値の 50% は保証する。
# （下限が低いと大半の個体が下限に張り付いて順位情報が消え、
# 選択が実質ランダム化して収束と適応度低下を招く）
MIN_ADJUSTMENT_FACTOR = 0.5


def _collect_gene_vectors(
    population: list[Any],
    gene_serializer: GeneSerializer,
    vectorize_gene: Callable[[StrategyGene], np.ndarray],
    on_error: Callable[[Exception], None] | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """
    個体群から有効な遺伝子ベクトルを収集する。

    遺伝子復元とベクトル化の共通処理を集約し、呼び出し側は
    キャッシュや後続処理に専念できるようにする。
    """
    vectors: list[np.ndarray] = []
    valid_indices: list[int] = []

    for i, individual in enumerate(population):
        try:
            gene = gene_serializer.from_list(individual, StrategyGene)
            if gene is None:
                continue

            vector = vectorize_gene(gene)
            vectors.append(vector)
            valid_indices.append(i)
        except Exception as e:
            if on_error is not None:
                on_error(e)

    return vectors, valid_indices


def silhouette_based_sharing(
    population: list[Any],
    gene_serializer: GeneSerializer,
    vectorize_gene: Callable[[StrategyGene], np.ndarray],
    objectives: Sequence[str] | None = None,
) -> list[Any]:
    """
    シルエットベースの共有を適用する。

    クラスタ中心に近い個体（シルエット小）ほど「集団内最悪値からの
    優位幅」を大きく縮小する。生値への乗算は最小化目的や負値では
    改善に働くため使わない（``shrink_advantage_toward_worst`` 参照）。
    """
    try:
        if len(population) <= 1:
            return population

        vectors, valid_indices = _collect_gene_vectors(
            population,
            gene_serializer=gene_serializer,
            vectorize_gene=vectorize_gene,
        )

        if len(vectors) < MIN_VECTORS_FOR_CLUSTERING:
            return population

        vectors_array = np.array(vectors)
        n_clusters = min(len(vectors_array) - 1, MAX_CLUSTERS)
        if n_clusters < 2:
            return population

        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
        labels = kmeans.fit_predict(vectors_array)

        silhouette_vals = silhouette_samples(vectors_array, labels)

        sample_values = next(
            (ind.fitness.values for ind in population if has_valid_fitness(ind)),
            (),
        )
        minimize_flags = resolve_minimize_flags(objectives, len(sample_values))
        worst_selection_values = compute_worst_selection_values(
            population, minimize_flags
        )

        for j, idx in enumerate(valid_indices):
            individual = population[idx]
            if has_valid_fitness(individual):
                silhouette_score = silhouette_vals[j]
                normalized_silhouette = (
                    silhouette_score + SILHOUETTE_OFFSET
                ) / SILHOUETTE_DIVISOR
                # クラスタ中心（normalized=1）ほど小さい倍率。下限は 0.5。
                span = 1.0 - MIN_ADJUSTMENT_FACTOR
                adjustment_factor = MIN_ADJUSTMENT_FACTOR + span * (
                    1.0 - normalized_silhouette
                )

                adjusted_values = tuple(
                    shrink_advantage_toward_worst(
                        fitness_val,
                        worst_selection_values[k]
                        if k < len(worst_selection_values)
                        else None,
                        minimize_flags[k] if k < len(minimize_flags) else False,
                        adjustment_factor,
                    )
                    for k, fitness_val in enumerate(individual.fitness.values)
                )
                individual.fitness.values = adjusted_values

        return population
    except Exception as e:
        logger.error(f"シルエットベース共有エラー: {e}")
        return population
