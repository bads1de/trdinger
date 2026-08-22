import pytest

from app.services.auto_strategy.config import objective_registry
from app.services.auto_strategy.config.constants import PENALTY_FITNESS_MAGNITUDE
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
            mode="walk_forward",
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
            mode="walk_forward",
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
            mode="walk_forward",
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


class TestPenaltyExclusionInAggregate:
    """制約違反ペナルティ値を含む集約のテスト"""

    def test_penalty_fitness_excluded_from_robust_median(self):
        """robust の中央値は合格foldのみから計算し、ワーストにペナルティを反映"""
        report = EvaluationReport.aggregate(
            mode="walk_forward",
            objectives=["weighted_score"],
            scenarios=[
                ScenarioEvaluation(
                    name="fold_0", fitness=(-PENALTY_FITNESS_MAGNITUDE,), passed=False
                ),
                ScenarioEvaluation(name="fold_1", fitness=(0.6,), passed=True),
                ScenarioEvaluation(name="fold_2", fitness=(0.8,), passed=True),
            ],
            aggregate_method="robust",
        )

        # 中央値は合格fold [0.6, 0.8] から 0.7。ワーストはペナルティを反映:
        # 0.7 * 0.7 + (-1e9) * 0.3
        expected = 0.7 * 0.7 + (-PENALTY_FITNESS_MAGNITUDE) * 0.3
        assert report.aggregated_fitness[0] == pytest.approx(expected)
        assert report.metadata["penalized_scenario_count"] == 1

    def test_penalty_fitness_excluded_from_mean_aggregation(self):
        report = EvaluationReport.aggregate(
            mode="walk_forward",
            objectives=["weighted_score"],
            scenarios=[
                ScenarioEvaluation(name="fold_0", fitness=(0.5,), passed=True),
                ScenarioEvaluation(name="fold_1", fitness=(0.7,), passed=True),
                ScenarioEvaluation(name="fold_2", fitness=(-1e9,), passed=False),
            ],
            aggregate_method="mean",
        )

        assert round(report.aggregated_fitness[0], 6) == 0.6
        assert report.metadata["penalized_scenario_count"] == 1

    def test_all_penalty_scenarios_aggregate_to_penalty(self):
        report = EvaluationReport.aggregate(
            mode="walk_forward",
            objectives=["weighted_score"],
            scenarios=[
                ScenarioEvaluation(name="fold_0", fitness=(-1e9,), passed=False),
                ScenarioEvaluation(name="fold_1", fitness=(-1e9,), passed=False),
            ],
            aggregate_method="robust",
        )

        # 全シナリオがペナルティの場合、方向を考慮したペナルティ値になる
        # （0.0 は最小化目的で最良スコアになるため使用しない）
        assert report.aggregated_fitness == (-PENALTY_FITNESS_MAGNITUDE,)
        assert report.metadata["penalized_scenario_count"] == 2

    def test_penalty_excluded_from_worst_case(self):
        report = EvaluationReport.aggregate(
            mode="walk_forward",
            objectives=["weighted_score"],
            scenarios=[
                ScenarioEvaluation(name="fold_0", fitness=(-1e9,), passed=False),
                ScenarioEvaluation(name="fold_1", fitness=(0.4,), passed=True),
            ],
            aggregate_method="robust",
        )

        assert report.primary_worst_case_fitness == 0.4

    def test_robust_worst_case_includes_penalty(self):
        """robust 集約のワーストケースはペナルティfoldを反映すること

        中央値は合格foldから計算するが、ワーストケースにペナルティ値
        （方向考慮）を反映するため、1つでも失敗foldがあると集約fitnessは
        大きく悪化する。
        """
        report = EvaluationReport.aggregate(
            mode="walk_forward",
            objectives=["weighted_score"],
            scenarios=[
                ScenarioEvaluation(name="fold_0", fitness=(0.4,), passed=True),
                ScenarioEvaluation(name="fold_1", fitness=(-1e9,), passed=False),
            ],
            aggregate_method="robust",
        )

        # 中央値 0.4 × 0.7 + ワースト（ペナルティ）× 0.3
        expected = 0.4 * 0.7 + (-PENALTY_FITNESS_MAGNITUDE) * 0.3
        assert report.aggregated_fitness[0] == pytest.approx(expected)
        # blend 後も有限値であり DB/JSON 保存可能な範囲に収まる
        assert abs(report.aggregated_fitness[0]) < PENALTY_FITNESS_MAGNITUDE

    def test_partial_failure_ranks_below_all_passing(self):
        """一部fold失敗の個体が全合格の個体に支配されないこと（回帰）

        旧実装はペナルティfoldを集計から完全除外するため、
        「5 fold中2失敗だが残りが高調子」な個体が
        「5 fold全部合格で低調」な個体に勝ってしまっていた。
        """
        partial_failure = EvaluationReport.aggregate(
            mode="walk_forward",
            objectives=["weighted_score"],
            scenarios=[
                ScenarioEvaluation(name="fold_0", fitness=(0.5,), passed=True),
                ScenarioEvaluation(name="fold_1", fitness=(0.6,), passed=True),
                ScenarioEvaluation(name="fold_2", fitness=(-1e9,), passed=False),
                ScenarioEvaluation(name="fold_3", fitness=(-1e9,), passed=False),
                ScenarioEvaluation(name="fold_4", fitness=(0.4,), passed=True),
            ],
            aggregate_method="robust",
        )
        all_passing = EvaluationReport.aggregate(
            mode="walk_forward",
            objectives=["weighted_score"],
            scenarios=[
                ScenarioEvaluation(name="fold_0", fitness=(0.3,), passed=True),
                ScenarioEvaluation(name="fold_1", fitness=(0.3,), passed=True),
                ScenarioEvaluation(name="fold_2", fitness=(0.3,), passed=True),
                ScenarioEvaluation(name="fold_3", fitness=(0.3,), passed=True),
                ScenarioEvaluation(name="fold_4", fitness=(0.3,), passed=True),
            ],
            aggregate_method="robust",
        )

        assert partial_failure.aggregated_fitness[0] < all_passing.aggregated_fitness[0]

    def test_robust_penalty_worst_works_for_minimize_objective(self):
        """最小化目的でも失敗foldがワーストケースに効くこと"""
        report = EvaluationReport.aggregate(
            mode="walk_forward",
            objectives=["max_drawdown"],
            scenarios=[
                ScenarioEvaluation(name="fold_0", fitness=(0.2,), passed=True),
                ScenarioEvaluation(
                    name="fold_1", fitness=(1e9,), passed=False
                ),  # 最小化のペナルティは +1e9
            ],
            aggregate_method="robust",
        )
        all_passing = EvaluationReport.aggregate(
            mode="walk_forward",
            objectives=["max_drawdown"],
            scenarios=[
                ScenarioEvaluation(name="fold_0", fitness=(0.2,), passed=True),
                ScenarioEvaluation(name="fold_1", fitness=(0.25,), passed=True),
            ],
            aggregate_method="robust",
        )

        # 最小化は小さいほど良い: 失敗入りは大きく悪化する
        assert report.aggregated_fitness[0] > all_passing.aggregated_fitness[0]

    def test_single_mode_with_penalty_only_aggregates_to_penalty(self):
        """single モードで唯一のシナリオがペナルティの場合、集約はペナルティ値になる"""
        report = EvaluationReport.aggregate(
            mode="single",
            objectives=["weighted_score"],
            scenarios=[
                ScenarioEvaluation(name="fold_0", fitness=(-1e9,), passed=False)
            ],
            aggregate_method="single",
        )

        assert report.aggregated_fitness == (-PENALTY_FITNESS_MAGNITUDE,)
        assert report.metadata["penalized_scenario_count"] == 1
