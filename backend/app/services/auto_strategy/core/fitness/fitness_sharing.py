"""
フィットネス共有（Fitness Sharing）

遺伝的アルゴリズムにおけるニッチ形成を実現するためのフィットネス共有機能。
類似した個体のフィットネス値を調整することで、多様な戦略の共存を促進します。
"""

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from app.services.auto_strategy.config.constants import OPERATORS
from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.serializers.serialization import GeneSerializer
from app.services.auto_strategy.utils.indicators import get_valid_indicator_types

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
from .fitness_sharing_similarity import (
    calculate_condition_similarity as _calculate_condition_similarity,
)
from .fitness_sharing_similarity import (
    calculate_indicator_similarity as _calculate_indicator_similarity,
)
from .fitness_sharing_similarity import (
    calculate_position_sizing_similarity as _calculate_position_sizing_similarity,
)
from .fitness_sharing_similarity import (
    calculate_risk_management_similarity as _calculate_risk_management_similarity,
)
from .fitness_sharing_similarity import calculate_similarity as _calculate_similarity
from .fitness_sharing_similarity import (
    calculate_tpsl_similarity as _calculate_tpsl_similarity,
)
from .fitness_sharing_similarity import check_none_similarity as _check_none_similarity
from .fitness_sharing_vectorizer import (
    _count_operand_types,
    _count_operators,
    build_behavior_profile,
)
from .fitness_sharing_vectorizer import vectorize_gene as _vectorize_gene
from .fitness_utils import has_valid_fitness

_FrozenKey = tuple | str | int | float | bool | None | bytes

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

    def __init__(
        self,
        sharing_radius: Optional[float] = None,
        alpha: Optional[float] = None,
        sampling_threshold: Optional[int] = None,
        sampling_ratio: Optional[float] = None,
        evaluation_report_provider: Optional[Callable[[Any], Any]] = None,
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
            self.indicator_types = get_valid_indicator_types()
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

    def apply_fitness_sharing(self, population: List[Any]) -> List[Any]:
        """
        個体群にフィットネス共有を適用（最適化版）

        ベクトル化とKD-Treeを使用してO(N²)からO(N log N)に計算量を削減。
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

            original_fitness: dict[int, tuple[float, ...]] = {}
            for i, individual in enumerate(population):
                if has_valid_fitness(individual):
                    original_fitness[i] = individual.fitness.values
                    shared_fitness_values = tuple(
                        float(
                            np.divide(
                                fitness_val,
                                niche_counts[i],
                                out=np.zeros_like(fitness_val),
                                where=niche_counts[i] != 0,
                            )
                        )
                        for fitness_val in individual.fitness.values
                    )
                    individual.fitness.values = shared_fitness_values

            result = _silhouette_based_sharing(
                population,
                gene_serializer=self.gene_serializer,
                vectorize_gene=resolve_vector,
            )

            # niche-count調整を元に戻し、シルエット調整のみを残す
            for i, individual in enumerate(population):
                if i in original_fitness and hasattr(individual, "fitness"):
                    silhouette_ratio = (
                        tuple(
                            float(np.divide(s, o, out=np.zeros_like(s), where=o != 0))
                            for s, o in zip(
                                individual.fitness.values, original_fitness[i]
                            )
                        )
                        if individual.fitness.valid
                        else original_fitness[i]
                    )
                    individual.fitness.values = tuple(
                        o * r for o, r in zip(original_fitness[i], silhouette_ratio)
                    )

            return result

        except Exception as e:
            logger.error(f"フィットネス共有適用エラー: {e}")
            return population

    def _get_feature_vector_cache_key_with_behavior(
        self,
        gene: StrategyGene,
        behavior_profile: Optional[Mapping[str, float]] = None,
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
        behavior_profile: Optional[Mapping[str, float]] = None,
    ) -> _FrozenKey:
        """互換性維持のための公開ラッパー。"""
        return self._get_feature_vector_cache_key_with_behavior(
            gene,
            behavior_profile=behavior_profile,
        )

    def set_evaluation_report_provider(
        self,
        provider: Optional[Callable[[Any], Any]],
    ) -> None:
        """behavior 特徴抽出に使う EvaluationReport 取得関数を設定する。"""
        self._evaluation_report_provider = provider

    def _get_evaluation_report(self, individual: Any) -> Optional[Any]:
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
        population: List[Any],
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
        behavior_profile: Optional[Mapping[str, float]] = None,
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

    def _calculate_similarity(self, gene1: StrategyGene, gene2: StrategyGene) -> float:
        """
        2つの戦略遺伝子間の類似度を計算
        """
        return _calculate_similarity(gene1, gene2)

    def _check_none_similarity(self, val1: Any, val2: Any) -> Optional[float]:
        """
        None値に対する類似度チェックの共通処理
        """
        return _check_none_similarity(val1, val2)

    def _calculate_indicator_similarity(
        self, indicators1: List[Any], indicators2: List[Any]
    ) -> float:
        """
        2つの指標セット間の類似度を計算
        """
        return _calculate_indicator_similarity(indicators1, indicators2)

    def _calculate_condition_similarity(
        self, conditions1: List[Any], conditions2: List[Any]
    ) -> float:
        """
        2つの条件リスト間の類似度を計算
        """
        return _calculate_condition_similarity(conditions1, conditions2)

    def _calculate_risk_management_similarity(
        self, risk1: Dict[str, Any], risk2: Dict[str, Any]
    ) -> float:
        """
        リスク管理設定の類似度を計算
        """
        return _calculate_risk_management_similarity(risk1, risk2)

    def _calculate_tpsl_similarity(self, tpsl1: Any, tpsl2: Any) -> float:
        """TP/SL遺伝子の類似度を計算"""
        return _calculate_tpsl_similarity(tpsl1, tpsl2)

    def _calculate_position_sizing_similarity(self, ps1: Any, ps2: Any) -> float:
        """ポジションサイジング遺伝子の類似度を計算"""
        return _calculate_position_sizing_similarity(ps1, ps2)

    def silhouette_based_sharing(self, population: List[Any]) -> List[Any]:
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
        )

    def _vectorize_gene(
        self,
        gene: StrategyGene,
        behavior_profile: Optional[Mapping[str, float]] = None,
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

    def _count_operators(self, conditions: List[Any], vector: np.ndarray) -> None:
        """条件リスト内のオペレータを再帰的にカウント"""
        _count_operators(conditions, self.operator_map, vector)

    def _count_operand_types(self, conditions: List[Any]) -> tuple[float, float]:
        """
        オペランドのタイプ（数値/動的）をカウント
        """
        return _count_operand_types(conditions)
