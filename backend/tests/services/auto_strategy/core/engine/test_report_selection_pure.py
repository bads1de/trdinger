from types import SimpleNamespace

from app.services.auto_strategy.core.engine.report_selection import (
    build_report_rank_key_from_primary_fitness,
    get_two_stage_best_individual,
    get_two_stage_rank,
    set_two_stage_metadata,
)
from app.services.auto_strategy.core.evaluation.evaluation_report import (
    EvaluationReport,
    ScenarioEvaluation,
)


def test_build_report_rank_key_from_primary_fitness_uses_report_values():
    report = EvaluationReport.aggregate(
        mode="robustness",
        objectives=["weighted_score"],
        scenarios=[
            ScenarioEvaluation(name="base", fitness=(0.9,), passed=True),
            ScenarioEvaluation(name="stress", fitness=(0.4,), passed=False),
        ],
        aggregate_method="robust",
    )

    key = build_report_rank_key_from_primary_fitness(
        1.5,
        report,
        min_pass_rate=0.75,
    )

    assert key == (0.0, 0.5, 0.4, 0.475)


class _SlottedIndividual:
    __slots__ = ("id", "fitness")

    def __init__(self, individual_id: str):
        self.id = individual_id
        self.fitness = SimpleNamespace(values=(0.0,))


def test_two_stage_metadata_is_readable_from_slotted_individual_fitness():
    leader = _SlottedIndividual("leader")
    other = _SlottedIndividual("other")

    set_two_stage_metadata(leader, rank=0, score=(1.0, 1.0, 0.8, 0.8))

    assert get_two_stage_rank(leader) == 0
    assert get_two_stage_best_individual([other, leader]) is leader
