"""
二段階選抜（Two-Stage Selection）モジュール

粗選抜（NSGA-II 等）後に評価レポートベースでエリートを再ランクし、
behavior 距離に基づいてエリートを多様化する選択ロジックを提供します。

``EvolutionRunner`` から分離された純粋な選択ロジックです。
呼び出し側は ``TwoStageSelection`` へ委譲します。
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

if TYPE_CHECKING:
    from ...config.ga_config import GAConfig
    from ..evaluation.evaluation_report import EvaluationReport

from .report_selection import (
    build_behavior_rank_key,
    build_report_rank_key,
    extract_primary_fitness,
    get_individual_identity,
    get_two_stage_elite_count,
    get_two_stage_pool_size,
    is_evaluation_report,
    merge_reranked_elites,
    set_two_stage_metadata,
)

logger = logging.getLogger(__name__)

# 二段階選抜で report が未取得/不合格の個体に使うデフォルト pass rate
DEFAULT_MIN_PASS_RATE = 0.0


class TwoStageSelection:
    """二段階選抜ロジック。

    ``EvolutionRunner`` から分離された選択ロジックをカプセル化します。
    必要とする外部依存（DEAPツールボックス・適応度共有・個別評価器）は
    コンストラクタで注入されます。
    """

    def __init__(
        self,
        toolbox: Any,
        fitness_sharing: Any | None = None,
        individual_evaluator: Any | None = None,
        behavior_profile_provider: Callable[[Any], Mapping[str, float] | None]
        | None = None,
    ):
        self.toolbox = toolbox
        self.fitness_sharing = fitness_sharing
        self.individual_evaluator = individual_evaluator
        # 並列評価時は report がメインプロセスに返らないため、
        # behavior summary を report の代替ランクキーとして使う
        self.behavior_profile_provider = behavior_profile_provider

    def apply_two_stage_selection(
        self,
        candidate_population: list[Any],
        population_size: int,
        config: GAConfig,
    ) -> list[Any]:
        """
        粗選抜後に report ベースでエリートを差し替える

        二段階選抜を適用します。まず通常の選択を行い、
        その後上位候補をreportベースで再ランクしてエリートを差し替えます。

        Args:
            candidate_population: 候補個体群
            population_size: 集団サイズ
            config: GA設定オブジェクト

        Returns:
            List[Any]: 選択された個体群

        Note:
            個別評価器がない場合は通常の選択のみ行います。
        """
        selected = list(self.toolbox.select(candidate_population, population_size))
        self._clear_two_stage_metadata(selected)

        elite_count = get_two_stage_elite_count(config, population_size)
        if elite_count <= 0 or self.individual_evaluator is None:
            return selected

        rerank_pool_size = get_two_stage_pool_size(
            len(candidate_population), elite_count, config
        )
        rerank_candidates = self.select_top_candidates(
            candidate_population,
            rerank_pool_size,
        )
        reranked_elites = self._select_report_ranked_elites(
            rerank_candidates,
            elite_count,
            config,
        )
        if not reranked_elites:
            return selected

        self._mark_two_stage_elites(reranked_elites)

        return merge_reranked_elites(selected, reranked_elites, population_size)

    def select_top_candidates(
        self,
        candidate_population: list[Any],
        limit: int,
    ) -> list[Any]:
        """
        主 fitness の上位候補を返す

        プライマリフィットネスに基づいて上位候補を選択します。

        Args:
            candidate_population: 候補個体群
            limit: 取得する候補数

        Returns:
            List[Any]: 上位候補リスト

        Note:
            limitが0以下の場合は空リストを返します。
        """
        if limit <= 0:
            return []
        if limit >= len(candidate_population):
            ranked = sorted(
                candidate_population,
                key=extract_primary_fitness,
                reverse=True,
            )
            return ranked

        ranked_top = heapq.nlargest(
            limit,
            enumerate(candidate_population),
            key=lambda item: (extract_primary_fitness(item[1]), -item[0]),
        )
        return [candidate for _, candidate in ranked_top]

    def _select_report_ranked_elites(
        self,
        candidates: list[Any],
        elite_count: int,
        config: GAConfig,
    ) -> list[tuple[Any, tuple[float, ...]]]:
        """
        候補を report ベースで再ランクしてエリートを返す

        評価レポートに基づいて候補を再ランクし、上位エリートを返します。

        Args:
            candidates: 候補リスト
            elite_count: エリート数
            config: GA設定オブジェクト

        Returns:
            List[tuple[Any, tuple[float, ...]]]: (個体, ランクキー)のタプルリスト

        Note:
            重複する候補はスキップされます。
        """
        ranked_candidates = []
        seen_keys = set()

        for candidate in candidates:
            candidate_key = get_individual_identity(candidate)
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)

            report = self._resolve_evaluation_report(candidate, config)
            two_stage_config = getattr(config, "two_stage_selection_config", None)
            min_pass_rate = getattr(
                two_stage_config, "min_pass_rate", DEFAULT_MIN_PASS_RATE
            )
            if report is not None:
                rank_key = build_report_rank_key(
                    candidate,
                    cast(Optional["EvaluationReport"], report),
                    min_pass_rate,
                )
            else:
                # report が取得できない場合（並列評価のデフォルト動作）、
                # behavior summary を report の代替として再ランクする。
                # サマリーも無い場合のみ fitness 単独のフォールバックになる。
                rank_key = build_behavior_rank_key(
                    extract_primary_fitness(candidate),
                    self._resolve_behavior_summary(candidate),
                    min_pass_rate=min_pass_rate,
                    primary_objective=self._primary_objective_name(config),
                )
            ranked_candidates.append((rank_key, candidate))

        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        ranked_candidates = self._diversify_reranked_elites(
            ranked_candidates,
            elite_count,
        )
        return [
            (candidate, rank_key)
            for rank_key, candidate in ranked_candidates[:elite_count]
        ]

    def _resolve_behavior_summary(self, candidate: Any) -> Mapping[str, float] | None:
        """並列評価キャッシュから候補の behavior summary を取得する。"""
        provider = self.behavior_profile_provider
        if not callable(provider):
            return None
        try:
            summary = provider(candidate)
        except Exception as e:
            logger.debug("behavior summary の取得に失敗しました: %s", e)
            return None
        return summary if isinstance(summary, Mapping) and summary else None

    @staticmethod
    def _primary_objective_name(config: GAConfig) -> str:
        """主目的関数名を返す（多目的・未設定の場合は空文字）。"""
        objectives = getattr(config, "objectives", None)
        if isinstance(objectives, (list, tuple)) and objectives:
            return str(objectives[0])
        return ""

    def _diversify_reranked_elites(
        self,
        ranked_candidates: list[tuple[tuple[float, ...], Any]],
        elite_count: int,
    ) -> list[tuple[tuple[float, ...], Any]]:
        """behavior ベースでエリート候補を greedy に散らす。"""
        if elite_count <= 1 or not ranked_candidates or self.fitness_sharing is None:
            return ranked_candidates

        build_vectors = getattr(
            self.fitness_sharing,
            "build_population_feature_vectors",
            None,
        )
        if not callable(build_vectors):
            return ranked_candidates

        try:
            candidate_vectors = cast(
                dict[int, np.ndarray],
                build_vectors([candidate for _, candidate in ranked_candidates]),
            )
        except Exception as exc:
            logger.debug("behavior diversity 用ベクトル取得に失敗しました: %s", exc)
            return ranked_candidates

        if len(candidate_vectors) < 2:
            return ranked_candidates

        distance_threshold = self._get_behavior_distance_threshold(candidate_vectors)
        if distance_threshold <= 0.0:
            return ranked_candidates

        diversified: list[tuple[tuple[float, ...], Any]] = []
        remaining = list(ranked_candidates)

        while remaining and len(diversified) < elite_count:
            chosen_index = 0
            if diversified:
                for index, (_, candidate) in enumerate(remaining):
                    if self._is_behaviorally_distinct(
                        candidate,
                        diversified,
                        candidate_vectors,
                        distance_threshold,
                    ):
                        chosen_index = index
                        break
            diversified.append(remaining.pop(chosen_index))

        diversified.extend(remaining)
        return diversified

    def _get_behavior_distance_threshold(
        self,
        candidate_vectors: dict[int, np.ndarray],
    ) -> float:
        """selection で使う behavior 距離の閾値を返す。"""
        if not candidate_vectors or self.fitness_sharing is None:
            return 0.0

        try:
            sharing_radius = float(getattr(self.fitness_sharing, "sharing_radius", 0.0))
        except (TypeError, ValueError):
            return 0.0

        if sharing_radius <= 0.0:
            return 0.0

        first_vector = next(iter(candidate_vectors.values()), None)
        if first_vector is None:
            return 0.0

        # 正規化済みベクトルでは値が一定の次元（使用されない指標など）は
        # すべて 0 になるため、レンジが 0 の次元を除外した実効次元数で
        # しきい値を計算する（niche 計算と同じ口径）。
        vectors_matrix = np.array(list(candidate_vectors.values()), dtype=float)
        active_dim_count = int(
            np.count_nonzero(vectors_matrix.max(axis=0) - vectors_matrix.min(axis=0))
        )
        if active_dim_count <= 0:
            return 0.0
        return sharing_radius * float(np.sqrt(active_dim_count))

    @staticmethod
    def _is_behaviorally_distinct(
        candidate: Any,
        selected: list[tuple[tuple[float, ...], Any]],
        candidate_vectors: dict[int, np.ndarray],
        distance_threshold: float,
    ) -> bool:
        """既に選ばれた個体群から十分離れているか判定する。"""
        candidate_vector = candidate_vectors.get(id(candidate))
        if candidate_vector is None:
            return True

        min_distance: float | None = None
        candidate_vector_array = np.asarray(candidate_vector, dtype=float)
        for _, selected_candidate in selected:
            selected_vector = candidate_vectors.get(id(selected_candidate))
            if selected_vector is None:
                continue
            distance = float(
                np.linalg.norm(
                    candidate_vector_array - np.asarray(selected_vector, dtype=float)
                )
            )
            min_distance = (
                distance
                if min_distance is None
                else min(
                    min_distance,
                    distance,
                )
            )

        if min_distance is None:
            return True
        return min_distance >= distance_threshold

    def _resolve_evaluation_report(
        self,
        candidate: object,
        config: GAConfig,
    ) -> object | None:
        """
        候補の report を取得し、必要なら主プロセスで再評価する

        候補の評価レポートを取得します。キャッシュにない場合は
        必要に応じて再評価を行います。

        Args:
            candidate: 候補個体
            config: GA設定オブジェクト

        Returns:
            Optional[object]: 評価レポート、取得失敗時はNone

        Note:
            個別評価器がない場合はNoneを返します。
        """
        if self.individual_evaluator is None:
            return None

        report: object | None = None
        get_cached_report = getattr(
            self.individual_evaluator,
            "get_cached_evaluation_report",
            None,
        )
        if report is None and callable(get_cached_report):
            report = get_cached_report(candidate)
            if not is_evaluation_report(report):
                report = None

        return report

    def _clear_two_stage_metadata(self, individuals: list[Any]) -> None:
        """
        前世代の二段階選抜メタデータをクリアする

        個体群から前世代の二段階選抜メタデータをクリアします。

        Args:
            individuals: 個体リスト

        Note:
            重複する個体は一度のみ処理されます。
        """
        seen_keys = set()
        for individual in individuals:
            candidate_key = get_individual_identity(individual)
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)
            self._set_two_stage_metadata(individual, None, None)

    def _mark_two_stage_elites(
        self,
        reranked_elites: list[tuple[Any, tuple[float, ...]]],
    ) -> None:
        """
        二段階選抜で確定したエリートへ順位を付与する

        再ランクされたエリートに順位とスコアのメタデータを設定します。

        Args:
            reranked_elites: (個体, ランクキー)のタプルリスト
        """
        for rank, (individual, score) in enumerate(reranked_elites):
            self._set_two_stage_metadata(individual, rank, score)

    def _set_two_stage_metadata(
        self,
        individual: Any,
        rank: int | None,
        score: Any | None,
    ) -> None:
        """
        個体へ二段階選抜のメタデータを付与する

        個体に二段階選抜の順位とスコアを設定します。

        Args:
            individual: 個体
            rank: 順位（オプション）
            score: スコア（オプション）

        Note:
            メタデータ設定に失敗した場合はログを出力してスキップします。
        """
        try:
            set_two_stage_metadata(individual, rank, score)
        except Exception as e:
            logger.debug("二段階選抜メタデータの設定をスキップしました: %s", e)
