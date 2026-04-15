import pytest

from app.services.auto_strategy.core.evaluation.evaluation_report import (
    EvaluationReport,
    ScenarioEvaluation,
)
from app.services.auto_strategy.core.evaluation.report_persistence import (
    attach_evaluation_summary,
    build_report_summary,
)


def test_build_report_summary_adds_selection_metadata():
    report = EvaluationReport.aggregate(
        mode="robustness",
        objectives=["weighted_score"],
        scenarios=[
            ScenarioEvaluation(name="base", fitness=(0.9,), passed=True),
            ScenarioEvaluation(name="symbol_ETHUSDT", fitness=(0.4,), passed=False),
        ],
        aggregate_method="robust",
    )

    summary = build_report_summary(
        report,
        selection_rank=0,
        selection_score=(1.0, 0.5, 0.4, 0.62),
        fitness_score=0.62,
    )

    assert summary["selection_rank"] == 0
    assert summary["fitness_score"] == 0.62
    assert summary["selection_components"] == {
        "pass_gate": 1.0,
        "pass_rate": 0.5,
        "worst_case": 0.4,
        "aggregated": 0.62,
    }
    assert summary["primary_worst_case_fitness"] == 0.4


def test_build_report_summary_keeps_extra_objective_components():
    report = EvaluationReport.aggregate(
        mode="robustness",
        objectives=["total_return", "max_drawdown"],
        scenarios=[
            ScenarioEvaluation(name="base", fitness=(0.9, 0.2), passed=True),
            ScenarioEvaluation(name="stress", fitness=(0.4, 0.5), passed=True),
        ],
        aggregate_method="robust",
    )

    summary = build_report_summary(
        report,
        selection_rank=1,
        selection_score=(1.0, 1.0, 0.4, 0.575, -0.5, -0.395),
        fitness_score=0.575,
    )

    selection_components = summary["selection_components"]
    assert selection_components["pass_gate"] == 1.0
    assert selection_components["pass_rate"] == 1.0
    assert selection_components["worst_case"] == pytest.approx(0.4)
    assert selection_components["aggregated"] == pytest.approx(0.575)
    assert len(selection_components["objective_components"]) == 1
    assert selection_components["objective_components"][0]["worst_case"] == pytest.approx(
        -0.5
    )
    assert selection_components["objective_components"][0]["aggregated"] == pytest.approx(
        -0.395
    )


def test_attach_evaluation_summary_merges_into_metadata():
    gene_data = {
        "id": "strategy-1",
        "indicators": [],
        "metadata": {"generated_by": "ga"},
    }

    merged = attach_evaluation_summary(
        gene_data,
        {
            "mode": "robustness",
            "pass_rate": 0.75,
        },
    )

    assert merged["metadata"]["generated_by"] == "ga"
    assert merged["metadata"]["evaluation_summary"] == {
        "mode": "robustness",
        "pass_rate": 0.75,
    }
