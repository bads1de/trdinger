"""
フィットネス共有のシルエットベース調整ユーティリティ
"""

import logging
from typing import Any, Callable, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.serializers.serialization import GeneSerializer

from .fitness_utils import has_valid_fitness

logger = logging.getLogger(__name__)

# 定数
MIN_VECTORS_FOR_CLUSTERING = 3
MAX_CLUSTERS = 3
RANDOM_STATE = 42
SILHOUETTE_OFFSET = 1.0
SILHOUETTE_DIVISOR = 2.0
MIN_ADJUSTMENT_FACTOR = 0.1


def _collect_gene_vectors(
    population: List[Any],
    gene_serializer: GeneSerializer,
    vectorize_gene: Callable[[StrategyGene], np.ndarray],
    on_error: Optional[Callable[[Exception], None]] = None,
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
    population: List[Any],
    gene_serializer: GeneSerializer,
    vectorize_gene: Callable[[StrategyGene], np.ndarray],
) -> List[Any]:
    """
    シルエットベースの共有を適用する。
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

        kmeans = KMeans(
            n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto"
        )
        labels = kmeans.fit_predict(vectors_array)

        silhouette_vals = silhouette_samples(vectors_array, labels)

        for j, idx in enumerate(valid_indices):
            individual = population[idx]
            if has_valid_fitness(individual):
                silhouette_score = silhouette_vals[j]
                normalized_silhouette = (
                    silhouette_score + SILHOUETTE_OFFSET
                ) / SILHOUETTE_DIVISOR
                adjustment_factor = max(
                    MIN_ADJUSTMENT_FACTOR, 1.0 - normalized_silhouette
                )

                original_fitness_values = individual.fitness.values
                adjusted_values = tuple(
                    fitness_val * adjustment_factor
                    for fitness_val in original_fitness_values
                )
                individual.fitness.values = adjusted_values

        return population
    except Exception as e:
        logger.error(f"シルエットベース共有エラー: {e}")
        return population
