"""
評価レポートベースの選抜ユーティリティ

二段階選抜で使用する共通ロジックを集約する。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from ..evaluation.evaluation_report import EvaluationReport

TWO_STAGE_RANK_ATTR = "_two_stage_selection_rank"
TWO_STAGE_SCORE_ATTR = "_two_stage_selection_score"
_DEFAULT_RERANK_MARGIN = 2
_MINIMIZE_OBJECTIVES = frozenset(
    {"max_drawdown", "ulcer_index", "trade_frequency_penalty"}
)


def get_two_stage_elite_count(config: Any, population_size: int) -> int:
    """二段階選抜に回すエリート数を返す。"""
    if (
        population_size <= 0
        or getattr(config, "enable_multi_objective", False)
        or not getattr(config, "enable_two_stage_selection", True)
    ):
        return 0

    elite_size = _safe_int(getattr(config, "elite_size", 0))
    configured_elite_count = _safe_int(
        getattr(config, "two_stage_elite_count", elite_size)
    )
    if elite_size <= 0 and configured_elite_count <= 0:
        return 0

    rerank_budget = configured_elite_count
    if rerank_budget <= 0:
        # 追加評価のコストを抑えるため、既存の小さいエリート予算を再利用する。
        rerank_budget = _safe_int(getattr(config, "tuning_elite_count", elite_size))
    if rerank_budget <= 0:
        rerank_budget = elite_size

    ceiling = elite_size if elite_size > 0 else rerank_budget
    return min(population_size, ceiling, rerank_budget)


def get_two_stage_pool_size(candidate_count: int, elite_count: int, config: Any) -> int:
    """二段階選抜で再ランクする候補数を返す。"""
    if candidate_count <= 0 or elite_count <= 0:
        return 0
    configured_pool_size = _safe_int(
        getattr(config, "two_stage_candidate_pool_size", elite_count + _DEFAULT_RERANK_MARGIN)
    )
    if configured_pool_size <= 0:
        configured_pool_size = elite_count + _DEFAULT_RERANK_MARGIN
    configured_pool_size = max(configured_pool_size, elite_count)
    return min(candidate_count, configured_pool_size)


def build_report_rank_key(
    individual: Any,
    report: Optional[EvaluationReport],
    min_pass_rate: float = 0.0,
) -> Tuple[float, float, float, float]:
    """report を利用した単一目的向けの再ランクキーを構築する。"""
    base_fitness = extract_primary_fitness(individual)
    return build_report_rank_key_from_primary_fitness(
        base_fitness,
        report,
        min_pass_rate=min_pass_rate,
    )


def build_report_rank_key_from_primary_fitness(
    primary_fitness: float,
    report: Optional[EvaluationReport],
    min_pass_rate: float = 0.0,
) -> Tuple[float, float, float, float]:
    """主 fitness 値と report から再ランクキーを構築する。"""
    base_fitness = float(primary_fitness)

    if not report or not report.aggregated_fitness:
        return (0.0, 0.0, base_fitness, base_fitness)

    objective = report.objectives[0] if report.objectives else ""
    is_minimize = objective in _MINIMIZE_OBJECTIVES
    scenario_values = [
        _to_selection_space(float(scenario.fitness[0]), is_minimize)
        for scenario in report.scenarios
        if scenario.fitness
    ]
    aggregated = _to_selection_space(float(report.aggregated_fitness[0]), is_minimize)
    worst_case = min(scenario_values) if scenario_values else aggregated
    pass_rate = float(report.pass_rate)
    pass_gate = 1.0 if pass_rate >= float(min_pass_rate) else 0.0

    return (
        pass_gate,
        pass_rate,
        worst_case,
        aggregated,
    )


def extract_primary_fitness(individual: Any) -> float:
    """個体から単一目的の主 fitness を取り出す。"""
    try:
        fitness = getattr(individual, "fitness", None)
        if fitness is not None:
            weighted_values = getattr(fitness, "wvalues", ())
            if isinstance(weighted_values, (tuple, list)) and weighted_values:
                return float(weighted_values[0])

            values = getattr(fitness, "values", ())
            if isinstance(values, (tuple, list)) and values:
                return float(values[0])
    except (TypeError, ValueError):
        pass
    return float("-inf")


def get_individual_identity(individual: Any) -> Any:
    """個体比較用の安定キーを返す。"""
    individual_id = getattr(individual, "id", None)
    return individual_id if individual_id not in (None, "") else id(individual)


def get_two_stage_rank(individual: Any) -> Optional[int]:
    """個体に付与された二段階選抜順位を返す。"""
    for target in _iter_two_stage_metadata_targets(individual):
        rank = getattr(target, TWO_STAGE_RANK_ATTR, None)
        if isinstance(rank, int):
            return int(rank)
    return None


def get_two_stage_score(individual: Any) -> Optional[Tuple[float, ...]]:
    """個体に付与された二段階選抜スコアを返す。"""
    for target in _iter_two_stage_metadata_targets(individual):
        score = getattr(target, TWO_STAGE_SCORE_ATTR, None)
        if isinstance(score, tuple):
            return score
        if isinstance(score, list):
            return tuple(float(value) for value in score)
    return None


def set_two_stage_metadata(
    individual: Any,
    rank: Optional[int],
    score: Optional[Any],
) -> None:
    """二段階選抜メタデータを fitness 側へ保存する。"""
    target = _get_two_stage_metadata_target(individual)
    try:
        setattr(target, TWO_STAGE_RANK_ATTR, rank)
        setattr(target, TWO_STAGE_SCORE_ATTR, score)
    except Exception:
        pass


def merge_reranked_elites(
    selected: Sequence[Any],
    reranked_elites: Sequence[tuple[Any, Any]],
    population_size: int,
) -> list[Any]:
    """昇格エリートを先頭へ反映しつつ集団サイズを維持する。"""
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


def is_evaluation_report(report: Any) -> bool:
    """EvaluationReport 互換のオブジェクトかを判定する。"""
    if not isinstance(report, EvaluationReport):
        return False
    return isinstance(report.aggregated_fitness, tuple)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_selection_space(value: float, is_minimize: bool) -> float:
    return -value if is_minimize else value


def _get_two_stage_metadata_target(individual: Any) -> Any:
    fitness = getattr(individual, "fitness", None)
    return fitness if fitness is not None else individual


def _iter_two_stage_metadata_targets(individual: Any) -> tuple[Any, ...]:
    primary = _get_two_stage_metadata_target(individual)
    if primary is individual:
        return (individual,)
    return (primary, individual)
