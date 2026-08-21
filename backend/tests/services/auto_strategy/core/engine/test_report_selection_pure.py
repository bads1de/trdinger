from types import SimpleNamespace

import pytest

from app.services.auto_strategy.config import objective_registry
from app.services.auto_strategy.core.engine.report_selection import (
    build_report_rank_key,
    build_report_rank_key_from_primary_fitness,
    get_individual_identity,
    get_two_stage_best_individual,
    get_two_stage_rank,
    is_evaluation_report,
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

    assert key == (0.0, 0.5, 0.4, 0.575)


def test_build_report_rank_key_from_primary_fitness_uses_objective_registry(
    monkeypatch,
):
    monkeypatch.setattr(
        objective_registry,
        "is_minimize_objective",
        lambda objective: objective == "custom_loss",
    )

    report = EvaluationReport.aggregate(
        mode="robustness",
        objectives=["custom_loss"],
        scenarios=[
            ScenarioEvaluation(name="base", fitness=(0.1,), passed=True),
            ScenarioEvaluation(name="stress", fitness=(0.5,), passed=False),
            ScenarioEvaluation(name="alt", fitness=(0.2,), passed=True),
        ],
        aggregate_method="robust",
    )

    key = build_report_rank_key_from_primary_fitness(
        0.7,
        report,
        min_pass_rate=0.5,
    )

    assert key[0] == 1.0
    assert key[1] == 2 / 3
    assert key[2] == -0.5
    assert round(key[3], 6) == -0.29


def test_build_report_rank_key_uses_all_objectives():
    report = EvaluationReport.aggregate(
        mode="robustness",
        objectives=["total_return", "max_drawdown"],
        scenarios=[
            ScenarioEvaluation(name="base", fitness=(0.9, 0.2), passed=True),
            ScenarioEvaluation(name="stress", fitness=(0.4, 0.5), passed=True),
        ],
        aggregate_method="robust",
    )
    individual = SimpleNamespace(fitness=SimpleNamespace(values=(1.5,)))

    key = build_report_rank_key(
        individual,
        report,
        min_pass_rate=0.5,
    )

    assert key == pytest.approx((1.0, 1.0, 0.4, 0.575, -0.5, -0.395))


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


class _CloneableGene:
    """id のみが異なる構造クローンを作れる最小遺伝子。"""

    def __init__(self, individual_id: str, indicator_type: str = "SMA"):
        self.id = individual_id
        self.indicators = [
            SimpleNamespace(
                type=indicator_type,
                timeframe=None,
                enabled=True,
                parameters={"length": 14},
            )
        ]
        self.long_entry_conditions = []
        self.short_entry_conditions = []
        self.long_exit_conditions = []
        self.short_exit_conditions = []
        self.stateful_conditions = []
        self.tool_genes = []
        self.risk_management = {}
        self.tpsl_gene = None
        self.long_tpsl_gene = None
        self.short_tpsl_gene = None
        self.position_sizing_gene = None
        self.entry_gene = None
        self.long_entry_gene = None
        self.short_entry_gene = None
        self.exit_gene = None


def test_identity_treats_structural_clones_as_same_individual():
    original = _CloneableGene("id-original")
    clone = _CloneableGene("id-clone")

    assert get_individual_identity(original) == get_individual_identity(clone)


def test_identity_distinguishes_different_structures():
    gene_a = _CloneableGene("id-a", indicator_type="SMA")
    gene_b = _CloneableGene("id-b", indicator_type="EMA")

    assert get_individual_identity(gene_a) != get_individual_identity(gene_b)


def test_identity_falls_back_to_id_attribute_on_signature_failure():
    # シグネチャ構築中に例外が発生するオブジェクトでは
    # id 属性へフォールバックする
    class _ExplodingGene:
        id = "fallback-id"

        @property
        def indicators(self):
            raise RuntimeError("signature failure")

    assert get_individual_identity(_ExplodingGene()) == "fallback-id"


def test_is_evaluation_report_distinguishes_report_and_plain_object():
    report = EvaluationReport.aggregate(
        mode="robustness",
        objectives=["weighted_score"],
        scenarios=[
            ScenarioEvaluation(name="base", fitness=(0.9,), passed=True),
        ],
        aggregate_method="single",
    )

    assert is_evaluation_report(report) is True
    assert is_evaluation_report(object()) is False
