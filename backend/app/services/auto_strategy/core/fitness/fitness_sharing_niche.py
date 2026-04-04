"""
フィットネス共有のニッチ計算ユーティリティ
"""

from typing import Sequence, cast

import numpy as np
from scipy.spatial import cKDTree


def compute_niche_counts_vectorized(
    vectors: np.ndarray,
    sharing_radius: float,
    sampling_threshold: int,
    sampling_ratio: float,
) -> np.ndarray:
    """
    ベクトル化されたニッチカウント計算（O(N log N)）
    """
    n_individuals = len(vectors)

    if n_individuals < 2:
        return np.ones(n_individuals)

    vectors_normalized = normalize_vectors(vectors)
    distance_threshold = sharing_radius * np.sqrt(vectors_normalized.shape[1])

    if n_individuals > sampling_threshold:
        return compute_niche_counts_sampling(
            vectors_normalized,
            distance_threshold=distance_threshold,
            sampling_ratio=sampling_ratio,
        )

    neighbors_list = find_neighbors_kdtree(
        vectors_normalized,
        radius=distance_threshold,
    )
    niche_counts = np.array(
        [max(1.0, float(len(neighbors))) for neighbors in neighbors_list]
    )
    return niche_counts


def find_neighbors_kdtree(
    vectors: np.ndarray,
    radius: float,
) -> Sequence[Sequence[int]]:
    """
    KD-Treeを使用して各点の近傍を探索する。
    """
    if len(vectors) < 1:
        return []

    tree = cKDTree(vectors)
    neighbors_list = tree.query_ball_point(vectors, r=radius)
    return cast(Sequence[Sequence[int]], neighbors_list)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    特徴ベクトルを正規化（0-1スケーリング）する。
    """
    if len(vectors) == 0:
        return vectors

    min_vals = vectors.min(axis=0)
    max_vals = vectors.max(axis=0)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    return (vectors - min_vals) / range_vals


def compute_niche_counts_sampling(
    vectors: np.ndarray,
    distance_threshold: float,
    sampling_ratio: float,
) -> np.ndarray:
    """
    サンプリングベースのニッチカウント近似（大規模集団用）
    """
    n_individuals = len(vectors)
    sample_size = max(10, int(n_individuals * sampling_ratio))

    rng_state = np.random.get_state()
    try:
        np.random.seed(42)
        sample_indices = np.random.choice(
            n_individuals, size=min(sample_size, n_individuals), replace=False
        )
        sample_vectors = vectors[sample_indices]

        sample_tree = cKDTree(sample_vectors)
        distances, _ = sample_tree.query(vectors, k=min(10, len(sample_indices)))
        distances = cast(np.ndarray, distances)

        if distances.ndim == 1:
            distances = distances.reshape(-1, 1)

        neighbors_in_sample = np.sum(distances < distance_threshold, axis=1)

        scale_factor = n_individuals / len(sample_indices)
        niche_counts = np.maximum(1.0, neighbors_in_sample * scale_factor)
        return niche_counts
    finally:
        np.random.set_state(rng_state)
