"""
フィットネス共有（Fitness Sharing）

遺伝的アルゴリズムにおけるニッチ形成を実現するためのフィットネス共有機能。
類似した個体のフィットネス値を調整することで、多様な戦略の共存を促進します。
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from app.services.auto_strategy.config.constants import OPERATORS
from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.serializers.serialization import GeneSerializer
from app.services.auto_strategy.utils.indicators import get_all_indicators
from app.types import SerializablePrimitive

from .fitness_sharing_niche import (
    compute_niche_counts_sampling as _compute_niche_counts_sampling,
)
from .fitness_sharing_niche import (
    compute_niche_counts_vectorized as _compute_niche_counts_vectorized,
)
from .fitness_sharing_niche import find_neighbors_kdtree as _find_neighbors_kdtree
from .fitness_sharing_niche import normalize_vectors as _normalize_vectors
from .fitness_sharing_silhouette import (
    _collect_gene_vectors,
)
from .fitness_sharing_silhouette import (
    silhouette_based_sharing as _silhouette_based_sharing,
)
from .fitness_sharing_vectorizer import build_behavior_profile
from .fitness_sharing_vectorizer import vectorize_gene as _vectorize_gene
from .fitness_validation import (
    compute_worst_selection_values,
    has_valid_fitness,
    resolve_minimize_flags,
    shrink_advantage_toward_worst,
)

_FrozenKey = tuple | bytes | SerializablePrimitive

logger = logging.getLogger(__name__)


class FitnessSharing:
    """
    フィットネス共有クラス

    個体間の類似度を計算し、類似した個体のフィットネス値を調整することで
    多様な戦略の共存を促進します。
    """

    # 定数
    DEFAULT_SAMPLING_THRESHOLD = 200
    SAMPLING_RATIO = 0.3
    DEFAULT_SHARING_RADIUS = 0.1
    DEFAULT_ALPHA = 1.0
    BEHAVIOR_SIGNATURE_PRECISION = 8
    # ニッチ除算の緩和指数。1.0 は古典的な共有（ niche_count で割る）だが、
    # 大きなニッチが一括して報酬を失い良好領域ごと消えるため sqrt で緩和する。
    NICHE_PENALTY_EXPONENT = 0.5

    def __init__(
        self,
        sharing_radius: float | None = None,
        alpha: float | None = None,
        sampling_threshold: int | None = None,
        sampling_ratio: float | None = None,
        evaluation_report_provider: Callable[[Any], Any] | None = None,
    ) -> None:
        """
        初期化

        Args:
            sharing_radius: 共有半径（類似度の閾値）
            alpha: 共有関数の形状パラメータ
            sampling_threshold: サンプリングを使用する集団サイズの閾値
            sampling_ratio: サンプリング時に使用するサンプル数の割合
        """
        if sharing_radius is None:
            sharing_radius = self.DEFAULT_SHARING_RADIUS
        if alpha is None:
            alpha = self.DEFAULT_ALPHA
        self.sharing_radius = sharing_radius
        self.alpha = alpha
        self.gene_serializer = GeneSerializer()
        self.sampling_threshold = (
            sampling_threshold
            if sampling_threshold is not None
            else self.DEFAULT_SAMPLING_THRESHOLD
        )
        self.sampling_ratio = (
            sampling_ratio if sampling_ratio is not None else self.SAMPLING_RATIO
        )
        self._feature_vector_cache: dict[_FrozenKey, np.ndarray] = {}
        self._evaluation_report_provider = evaluation_report_provider

        # 指標タイプマップの初期化（ベクトル化用）
        try:
            self.indicator_types = get_all_indicators()
            self.indicator_types.sort()
            self.indicator_map = {
                name: i for i, name in enumerate(self.indicator_types)
            }
        except Exception as e:
            logger.warning(f"指標タイプ取得失敗: {e}")
            self.indicator_types = []
            self.indicator_map = {}

        # オペレータマップの初期化
        try:
            self.operator_types = OPERATORS.copy()
            self.operator_types.extend(["AND", "OR"])
            self.operator_types.sort()
            self.operator_map = {op: i for i, op in enumerate(self.operator_types)}
        except Exception as e:
            logger.warning(f"オペレータタイプ取得失敗: {e}")
            self.operator_types = []
            self.operator_map = {}

    def apply_fitness_sharing(
        self, population: list[Any], objectives: Sequence[str] | None = None
    ) -> list[Any]:
        """
        個体群にフィットネス共有を適用（最適化版）

        ベクトル化とKD-Treeを使用してO(N²)からO(N log N)に計算量を削減。

        混雑した個体の適応度は「集団内最悪値からの優位幅」をニッチカウントで
        縮小する形で減衰させる（``shrink_advantage_toward_worst`` 参照）。
        生値の除算は最小化目的や負値で「改善」に働くため使わない。

        Args:
            population: 対象個体群。
            objectives: 目的関数名列。方向（最小化/最大化）の判定に使う。
                None の場合はすべて最大化として扱う。

        Note:
            適応度値はニッチカウントとシルエット係数で「一時的に」調整される。
            選択以外の用途（統計・HoF・次世代の評価値）に影響させないため、
            呼び出し側は選択完了後に必ず元の値へ復元すること。
        """
        try:
            if len(population) <= 1:
                return population

            behavior_profiles = self._build_behavior_profile_map(population)

            def resolve_vector(gene: StrategyGene) -> np.ndarray:
                behavior_profile = behavior_profiles.get(id(gene))
                return self._resolve_feature_vector(
                    gene,
                    behavior_profile=behavior_profile,
                )

            vectors, valid_indices = _collect_gene_vectors(
                population,
                gene_serializer=self.gene_serializer,
                vectorize_gene=resolve_vector,
                on_error=lambda e: logger.warning(f"個体の処理に失敗: {e}"),
            )

            if len(vectors) < 2:
                return population

            max_dim = max(v.shape[0] for v in vectors if isinstance(v, np.ndarray))
            vectors_padded: list[np.ndarray] = []
            for v in vectors:
                if v.shape[0] < max_dim:
                    padding = np.zeros(max_dim - v.shape[0])
                    vectors_padded.append(np.concatenate([v, padding]))
                else:
                    vectors_padded.append(v)

            vectors_array = np.array(vectors_padded)

            niche_counts_vectorized = self.compute_niche_counts_vectorized(
                vectors_array
            )

            niche_counts = [1.0] * len(population)
            for idx, valid_idx in enumerate(valid_indices):
                niche_counts[valid_idx] = niche_counts_vectorized[idx]

            # 適応度はニッチカウントとシルエット係数で一時的に調整される。
            # 呼び出し側（EvolutionRunner）は選択完了後に元の値へ復元するため、
            # ここでは復元を行わない（世代をまたいだ減衰を防ぐ）。
            # 減衰は目的の方向と値の符号に依存しない「優位幅の縮小」で行い、
            # ニッチ除算は sqrt で緩和する（定数のdocstring参照）。
            sample_values = next(
                (ind.fitness.values for ind in population if has_valid_fitness(ind)),
                (),
            )
            minimize_flags = resolve_minimize_flags(objectives, len(sample_values))
            worst_selection_values = compute_worst_selection_values(
                population, minimize_flags
            )

            for i, individual in enumerate(population):
                if has_valid_fitness(individual):
                    soften_niche_count = float(
                        np.power(max(niche_counts[i], 1.0), self.NICHE_PENALTY_EXPONENT)
                    )
                    shared_fitness_values = tuple(
                        shrink_advantage_toward_worst(
                            fitness_val,
                            worst_selection_values[j]
                            if j < len(worst_selection_values)
                            else None,
                            minimize_flags[j] if j < len(minimize_flags) else False,
                            1.0 / soften_niche_count,
                        )
                        for j, fitness_val in enumerate(individual.fitness.values)
                    )
                    individual.fitness.values = shared_fitness_values

            return _silhouette_based_sharing(
                population,
                gene_serializer=self.gene_serializer,
                vectorize_gene=resolve_vector,
                objectives=objectives,
            )

        except Exception as e:
            logger.error(f"フィットネス共有適用エラー: {e}")
            return population

    def _get_feature_vector_cache_key_with_behavior(
        self,
        gene: StrategyGene,
        behavior_profile: Mapping[str, float] | None = None,
    ) -> _FrozenKey:
        """behavior profile も含めた特徴ベクトルキャッシュキーを生成する。"""
        try:
            base_key = self.gene_serializer._generate_cache_key(gene)
            if not behavior_profile:
                return base_key

            behavior_signature = tuple(
                (
                    key,
                    round(float(value), self.BEHAVIOR_SIGNATURE_PRECISION),
                )
                for key, value in sorted(behavior_profile.items())
            )
            return base_key, behavior_signature
        except Exception as e:
            logger.debug(f"特徴ベクトルキャッシュキーの生成に失敗しました: {e}")
            return str(id(gene))

    def _get_feature_vector_cache_key(
        self,
        gene: StrategyGene,
        behavior_profile: Mapping[str, float] | None = None,
    ) -> _FrozenKey:
        """互換性維持のための公開ラッパー。"""
        return self._get_feature_vector_cache_key_with_behavior(
            gene,
            behavior_profile=behavior_profile,
        )

    def set_evaluation_report_provider(
        self,
        provider: Callable[[Any], Any] | None,
    ) -> None:
        """behavior 特徴抽出に使う EvaluationReport 取得関数を設定する。"""
        self._evaluation_report_provider = provider

    def _get_evaluation_report(self, individual: Any) -> Any | None:
        """評価レポート取得関数を安全に呼び出す。"""
        if not callable(self._evaluation_report_provider):
            return None
        try:
            return self._evaluation_report_provider(individual)
        except Exception as e:
            logger.debug("evaluation report の取得に失敗しました: %s", e)
            return None

    def _build_behavior_profile(self, individual: Any) -> dict[str, float]:
        """個体の評価結果から behavior profile を構築する。"""
        fitness = getattr(individual, "fitness", None)
        fitness_values = getattr(fitness, "values", None)
        report = self._get_evaluation_report(individual)
        return build_behavior_profile(
            fitness_values=fitness_values,
            evaluation_report=report,
        )

    def _build_behavior_profile_map(
        self,
        population: list[Any],
    ) -> dict[int, dict[str, float]]:
        """個体ごとの behavior profile を事前計算する。"""
        profiles: dict[int, dict[str, float]] = {}
        for individual in population:
            try:
                gene = self.gene_serializer.from_list(individual, StrategyGene)
                if gene is None:
                    continue
                profiles[id(gene)] = self._build_behavior_profile(individual)
            except Exception as e:
                logger.debug("behavior profile 構築に失敗しました: %s", e)
        return profiles

    def _resolve_feature_vector(
        self,
        gene: StrategyGene,
        behavior_profile: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """gene と behavior profile から特徴ベクトルを取得する。"""
        cache_key = self._get_feature_vector_cache_key(
            gene,
            behavior_profile=behavior_profile,
        )
        vector = self._feature_vector_cache.get(cache_key)
        if vector is None:
            vector = self._vectorize_gene(
                gene,
                behavior_profile=behavior_profile,
            )
            self._feature_vector_cache[cache_key] = vector
        return vector

    def build_population_feature_vectors(
        self,
        population: Sequence[Any],
    ) -> dict[int, np.ndarray]:
        """個体群を selection 用の正規化ベクトルへ変換する。"""
        if not population:
            return {}

        behavior_profiles = self._build_behavior_profile_map(list(population))
        raw_vectors: list[np.ndarray] = []
        vector_keys: list[int] = []

        for individual in population:
            try:
                gene = self.gene_serializer.from_list(individual, StrategyGene)
                if gene is None:
                    continue
                behavior_profile = behavior_profiles.get(id(gene))
                raw_vectors.append(
                    self._resolve_feature_vector(
                        gene,
                        behavior_profile=behavior_profile,
                    )
                )
                vector_keys.append(id(individual))
            except Exception as e:
                logger.debug("selection 用ベクトル構築に失敗しました: %s", e)

        if not raw_vectors:
            return {}

        max_dim = max(vector.shape[0] for vector in raw_vectors)
        padded_vectors: list[np.ndarray] = []
        for vector in raw_vectors:
            if vector.shape[0] < max_dim:
                padding = np.zeros(max_dim - vector.shape[0])
                padded_vectors.append(np.concatenate([vector, padding]))
            else:
                padded_vectors.append(vector)

        normalized_vectors = self._normalize_vectors(np.array(padded_vectors))
        return {
            vector_keys[index]: normalized_vectors[index].copy()
            for index in range(len(vector_keys))
        }

    def compute_niche_counts_vectorized(self, vectors: np.ndarray) -> np.ndarray:
        """
        ベクトル化されたニッチカウント計算（O(N log N)）
        """
        return _compute_niche_counts_vectorized(
            vectors,
            sharing_radius=self.sharing_radius,
            sampling_threshold=self.sampling_threshold,
            sampling_ratio=self.sampling_ratio,
        )

    def find_neighbors_kdtree(
        self, vectors: np.ndarray, radius: float
    ) -> Sequence[Sequence[int]]:
        """
        KD-Treeを使用して各点の近傍を探索（O(N log N)）
        """
        return _find_neighbors_kdtree(vectors, radius)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        特徴ベクトルを正規化（0-1スケーリング）
        """
        return _normalize_vectors(vectors)

    def _compute_niche_counts_sampling(
        self, vectors: np.ndarray, distance_threshold: float
    ) -> np.ndarray:
        """
        サンプリングベースのニッチカウント近似（大規模集団用）
        """
        return _compute_niche_counts_sampling(
            vectors,
            distance_threshold=distance_threshold,
            sampling_ratio=self.sampling_ratio,
        )

    def silhouette_based_sharing(
        self, population: list[Any], objectives: Sequence[str] | None = None
    ) -> list[Any]:
        """
        シルエットベースの共有
        """
        behavior_profiles = self._build_behavior_profile_map(population)

        def resolve_vector(gene: StrategyGene) -> np.ndarray:
            behavior_profile = behavior_profiles.get(id(gene))
            return self._vectorize_gene(gene, behavior_profile=behavior_profile)

        return _silhouette_based_sharing(
            population,
            gene_serializer=self.gene_serializer,
            vectorize_gene=resolve_vector,
            objectives=objectives,
        )

    def _vectorize_gene(
        self,
        gene: StrategyGene,
        behavior_profile: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """
        StrategyGeneを数値的な特徴ベクトルに変換します。
        """
        return _vectorize_gene(
            gene,
            indicator_types=self.indicator_types,
            indicator_map=self.indicator_map,
            operator_types=self.operator_types,
            operator_map=self.operator_map,
            behavior_profile=behavior_profile,
        )
