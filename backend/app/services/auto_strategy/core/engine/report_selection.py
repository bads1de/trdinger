"""
評価レポートベースの選抜ユーティリティ

二段階選抜で使用する共通ロジックを集約する。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple, TypeGuard

from app.services.auto_strategy.config import objective_registry

from ..evaluation.evaluation_report import EvaluationReport
from .fitness_utils import (
    extract_individual_primary_fitness as _extract_individual_primary_fitness,
)

logger = logging.getLogger(__name__)

TWO_STAGE_RANK_ATTR = "_two_stage_selection_rank"
TWO_STAGE_SCORE_ATTR = "_two_stage_selection_score"
_DEFAULT_RERANK_MARGIN = 2


def get_two_stage_elite_count(config: object, population_size: int) -> int:
    """二段階選抜に回すエリート数を返す。

    GA設定と集団サイズから、二段階選抜（Two-Stage Selection）で
    エリートとして直接選抜される個体数を計算します。

    Args:
        config: GA設定オブジェクト。two_stage_selection_config、elite_sizeなどを含む。
        population_size: 現在の集団サイズ。

    Returns:
        int: エリートとして直接選抜される個体数。
            無効な設定の場合は0。
    """
    two_stage_config = getattr(config, "two_stage_selection_config", None)
    if population_size <= 0 or not getattr(two_stage_config, "enabled", True):
        return 0

    elite_size = _safe_int(getattr(config, "elite_size", 0))
    configured_elite_count = _safe_int(
        getattr(two_stage_config, "elite_count", elite_size)
    )
    if elite_size <= 0 and configured_elite_count <= 0:
        return 0

    rerank_budget = configured_elite_count
    if rerank_budget <= 0:
        rerank_budget = elite_size
    if rerank_budget <= 0:
        rerank_budget = elite_size

    ceiling = elite_size if elite_size > 0 else rerank_budget
    return min(population_size, ceiling, rerank_budget)


def get_two_stage_pool_size(candidate_count: int, elite_count: int, config: object) -> int:
    """二段階選抜で再ランクする候補数を返す。

    エリート数と設定から、再ランク（再評価）対象となる候補数を計算します。

    Args:
        candidate_count: 候補となる個体数。
        elite_count: エリートとして選抜された個体数。
        config: GA設定オブジェクト。two_stage_selection_config.candidate_pool_sizeを含む。

    Returns:
        int: 再ランク対象となる個体数。
            無効な設定の場合は0。
    """
    if candidate_count <= 0 or elite_count <= 0:
        return 0
    two_stage_config = getattr(config, "two_stage_selection_config", None)
    configured_pool_size = _safe_int(
        getattr(
            two_stage_config,
            "candidate_pool_size",
            elite_count + _DEFAULT_RERANK_MARGIN,
        )
    )
    if configured_pool_size <= 0:
        configured_pool_size = elite_count + _DEFAULT_RERANK_MARGIN
    configured_pool_size = max(configured_pool_size, elite_count)
    return min(candidate_count, configured_pool_size)


def build_report_rank_key(
    individual: object,
    report: Optional["EvaluationReport"],
    min_pass_rate: float = 0.0,
) -> Tuple[float, ...]:
    """report を利用した再ランクキーを構築する。

    評価レポートの合格率と個体のフィットネス値を組み合わせて、
    二段階選抜用のソートキーを生成します。

    Args:
        individual: 評価対象の個体。
        report: 評価レポート（オプション）。Noneの場合はフィットネス値のみ使用。
        min_pass_rate: 最低合格率。これ未満の個体はソートで不利になる。

    Returns:
        Tuple[float, ...]: 再ランク用のソートキータプル。
            値が小さいほど上位に来る（ソート昇順で上位）。
    """
    base_fitness = extract_primary_fitness(individual)
    return build_report_rank_key_from_primary_fitness(
        base_fitness,
        report,
        min_pass_rate=min_pass_rate,
    )


def build_report_rank_key_from_primary_fitness(
    primary_fitness: float,
    report: Optional["EvaluationReport"],
    min_pass_rate: float = 0.0,
) -> Tuple[float, ...]:
    """主fitness値とreportから再ランクキーを構築する。

    評価レポートの合格率、ワーストケーススコア、集約フィットネスなどを
    組み合わせて、二段階選抜用のソートキーを生成します。
    多目的の場合は目的関数の順にスコアを連結した可変長タプルを返します。

    Args:
        primary_fitness: 個体の主フィットネス値。
        report: 評価レポート（オプション）。
        min_pass_rate: 最低合格率。

    Returns:
        Tuple[float, ...]: 再ランク用のソートキータプル。
            単一目的では (pass_gate, pass_rate, worst_case, aggregated) の順、
            多目的では (pass_gate, pass_rate, worst_case_1, aggregated_1, ...) の順で、
            値が小さいほど上位に来る。
    """
    base_fitness = float(primary_fitness)

    if not report or not report.aggregated_fitness:
        return (0.0, 0.0, base_fitness, base_fitness)

    pass_rate = float(report.pass_rate)
    pass_gate = 1.0 if pass_rate >= float(min_pass_rate) else 0.0

    objective_count = len(report.aggregated_fitness)
    if objective_count <= 1:
        objective = report.objectives[0] if report.objectives else ""
        scenario_values = [
            objective_registry.to_selection_space(float(scenario.fitness[0]), objective)
            for scenario in report.scenarios
            if scenario.fitness
        ]
        aggregated = objective_registry.to_selection_space(
            float(report.aggregated_fitness[0]),
            objective,
        )
        worst_case = min(scenario_values) if scenario_values else aggregated
        return (
            pass_gate,
            pass_rate,
            worst_case,
            aggregated,
        )

    rank_components: list[float] = [pass_gate, pass_rate]
    for index in range(objective_count):
        objective = report.objectives[index] if index < len(report.objectives) else ""
        scenario_values = [
            objective_registry.to_selection_space(
                float(scenario.fitness[index]), objective
            )
            for scenario in report.scenarios
            if scenario.fitness and len(scenario.fitness) > index
        ]
        aggregated = objective_registry.to_selection_space(
            float(report.aggregated_fitness[index]),
            objective,
        )
        worst_case = min(scenario_values) if scenario_values else aggregated
        rank_components.extend((worst_case, aggregated))

    return tuple(rank_components)


def extract_primary_fitness(individual: object) -> float:
    """個体から主 fitness を取り出す。"""
    return _extract_individual_primary_fitness(individual)


def get_individual_identity(individual: object) -> object:
    """個体比較用の安定キーを返す。

    個体のid属性を優先して使用し、存在しない場合は
    PythonのオブジェクトIDをフォールバックとして使用します。

    Args:
        individual: 比較対象の個体。

    Returns:
        個体を一意に識別するキー（文字列のidまたはintのオブジェクトID）。
    """
    individual_id = getattr(individual, "id", None)
    return individual_id if individual_id not in (None, "") else id(individual)


def get_two_stage_rank(individual: object) -> Optional[int]:
    """個体に付与された二段階選抜順位を返す。"""
    for target in _iter_two_stage_metadata_targets(individual):
        rank = getattr(target, TWO_STAGE_RANK_ATTR, None)
        if isinstance(rank, int):
            return int(rank)
    return None


def get_two_stage_score(individual: object) -> Optional[Tuple[float, ...]]:
    """個体に付与された二段階選抜スコアを返す。"""
    for target in _iter_two_stage_metadata_targets(individual):
        score = getattr(target, TWO_STAGE_SCORE_ATTR, None)
        if isinstance(score, tuple):
            return score
        if isinstance(score, list):
            return tuple(float(value) for value in score)
    return None


def set_two_stage_metadata(
    individual: object,
    rank: Optional[int],
    score: Optional[object],
) -> None:
    """二段階選抜メタデータを fitness 側へ保存する。"""
    target = _get_two_stage_metadata_target(individual)
    try:
        setattr(target, TWO_STAGE_RANK_ATTR, rank)
        setattr(target, TWO_STAGE_SCORE_ATTR, score)
    except Exception as e:
        logger.debug(f"二段階選抜メタデータの設定に失敗しました: {e}")
        pass


def merge_reranked_elites(
    selected: Sequence[Any],
    reranked_elites: Sequence[tuple[Any, Any]],
    population_size: int,
) -> list[Any]:
    """昇格エリートを先頭へ反映しつつ集団サイズを維持する。

    再ランクされたエリート個体を先頭に配置し、残りの個体を
    後ろに続けることで、新しい集団を構築します。
    重複する個体は除去され、集団サイズが維持されます。

    Args:
        selected: 一段階目で選抜された個体のシーケンス。
        reranked_elites: 再ランクされたエリート個体のリスト。
            各要素は（個体、スコア）のタプル。
        population_size: 目標とする集団サイズ。

    Returns:
        list[Any]: マージ後の新しい集団。population_size以下に切り詰められる。
    """
    remaining = list(selected)
    for elite, _ in reranked_elites:
        elite_key = get_individual_identity(elite)
        for index, individual in enumerate(remaining):
            if get_individual_identity(individual) == elite_key:
                del remaining[index]
                break

    merged = [individual for individual, _ in reranked_elites] + remaining
    return merged[:population_size]


def get_two_stage_best_individual(population: Sequence[Any]) -> Optional[Any]:
    """二段階選抜で先頭に選ばれた個体を返す。"""
    ranked = []
    for individual in population:
        rank = get_two_stage_rank(individual)
        if rank is None:
            continue
        ranked.append((rank, individual))

    if not ranked:
        return None

    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def is_evaluation_report(report: object) -> TypeGuard[EvaluationReport]:
    """EvaluationReport 互換のオブジェクトかを判定する。"""
    if not isinstance(report, EvaluationReport):
        return False
    return isinstance(report.aggregated_fitness, tuple)


def _safe_int(value: Any) -> int:
    """値を安全に整数に変換する。

    変換できない場合は0を返す。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _get_two_stage_metadata_target(individual: object) -> object:
    """2段階評価のメタデータを取得する対象を抽出する。

    個体にfitness属性があればそれを返し、なければ個体自身を返す。
    """
    fitness = getattr(individual, "fitness", None)
    return fitness if fitness is not None else individual


def _iter_two_stage_metadata_targets(individual: object) -> tuple[object, ...]:
    primary = _get_two_stage_metadata_target(individual)
    if primary is individual:
        return (individual,)
    return (primary, individual)
