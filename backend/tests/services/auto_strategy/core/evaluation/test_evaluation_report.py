from app.services.auto_strategy.config import objective_registry
from app.services.auto_strategy.core.evaluation.evaluation_report import (
    EvaluationReport,
    ScenarioEvaluation,
)


class TestEvaluationReport:
    def test_aggregate_robust_for_maximize_objective(self):
        report = EvaluationReport.aggregate(
            mode="walk_forward",
            objectives=["weighted_score"],
            scenarios=[
                ScenarioEvaluation(name="fold_1", fitness=(1.0,), passed=True),
                ScenarioEvaluation(name="fold_2", fitness=(0.2,), passed=False),
                ScenarioEvaluation(name="fold_3", fitness=(0.8,), passed=True),
            ],
            aggregate_method="robust",
        )

        assert round(report.aggregated_fitness[0], 6) == 0.62
        assert report.pass_rate == 2 / 3

    def test_aggregate_robust_for_minimize_objective(self):
        report = EvaluationReport.aggregate(
            mode="purged_kfold",
            objectives=["max_drawdown"],
            scenarios=[
                ScenarioEvaluation(name="fold_1", fitness=(0.1,), passed=True),
                ScenarioEvaluation(name="fold_2", fitness=(0.2,), passed=True),
                ScenarioEvaluation(name="fold_3", fitness=(0.5,), passed=False),
            ],
            aggregate_method="robust",
        )

        assert round(report.aggregated_fitness[0], 6) == 0.29
        assert report.pass_rate == 2 / 3

    def test_aggregate_robust_uses_objective_registry(self, monkeypatch):
        monkeypatch.setattr(
            objective_registry,
            "is_minimize_objective",
            lambda objective: objective == "custom_loss",
        )

        report = EvaluationReport.aggregate(
            mode="purged_kfold",
            objectives=["custom_loss"],
            scenarios=[
                ScenarioEvaluation(name="fold_1", fitness=(0.1,), passed=True),
                ScenarioEvaluation(name="fold_2", fitness=(0.2,), passed=True),
                ScenarioEvaluation(name="fold_3", fitness=(0.5,), passed=False),
            ],
            aggregate_method="robust",
        )

        assert round(report.aggregated_fitness[0], 6) == 0.29
        assert report.primary_worst_case_fitness == 0.5

    def test_aggregate_weighted_uses_weights(self):
        report = EvaluationReport.aggregate(
            mode="oos",
            objectives=["weighted_score"],
            scenarios=[
                ScenarioEvaluation(name="is", fitness=(1.0,), passed=True),
                ScenarioEvaluation(name="oos", fitness=(0.4,), passed=True),
            ],
            aggregate_method="weighted",
            weights=[0.25, 0.75],
        )

        assert round(report.aggregated_fitness[0], 6) == 0.55
        assert report.pass_rate == 1.0

    def test_to_summary_dict_includes_worst_case_and_scenarios(self):
        report = EvaluationReport.aggregate(
            mode="robustness",
            objectives=["weighted_score"],
            scenarios=[
                ScenarioEvaluation(name="base", fitness=(0.9,), passed=True),
                ScenarioEvaluation(name="symbol_ETHUSDT", fitness=(0.4,), passed=False),
                ScenarioEvaluation(name="slippage_0.0015", fitness=(0.7,), passed=True),
            ],
            aggregate_method="robust",
            metadata={"source": "two_stage_selection"},
        )

        summary = report.to_summary_dict()

        assert summary["mode"] == "robustness"
        assert summary["primary_objective"] == "weighted_score"
        assert summary["primary_aggregated_fitness"] == report.aggregated_fitness[0]
        assert summary["primary_worst_case_fitness"] == 0.4
        assert summary["scenario_count"] == 3
        assert summary["metadata"]["source"] == "two_stage_selection"
        assert summary["metadata"]["scenario_count"] == 3
        assert summary["scenarios"][1]["name"] == "symbol_ETHUSDT"
        assert summary["scenarios"][1]["passed"] is False
        assert summary["scenarios"][1]["fitness"] == [0.4]
        assert summary["scenarios"][1]["primary_fitness"] == 0.4
