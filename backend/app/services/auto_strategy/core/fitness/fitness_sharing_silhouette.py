"""
フィットネス共有のシルエットベース調整ユーティリティ
"""

import logging
from typing import Any, Callable, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.serializers.serialization import GeneSerializer

logger = logging.getLogger(__name__)


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

        vectors = []
        valid_indices = []
        for i, individual in enumerate(population):
            try:
                gene = gene_serializer.from_list(individual, StrategyGene)
                if gene is not None:
                    vector = vectorize_gene(gene)
                    vectors.append(vector)
                    valid_indices.append(i)
            except Exception:
                continue

        if len(vectors) <= 1:
            return population

        vectors_array = np.array(vectors)
        n_clusters = min(len(vectors_array), 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(vectors_array)

        silhouette_vals = silhouette_samples(vectors_array, labels)

        for j, idx in enumerate(valid_indices):
            individual = population[idx]
            if hasattr(individual, "fitness") and individual.fitness.valid:
                silhouette_score = silhouette_vals[j]
                normalized_silhouette = (silhouette_score + 1.0) / 2.0
                adjustment_factor = max(0.1, 1.0 - normalized_silhouette)

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
