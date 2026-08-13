"""
自動検証パイプライン（StrategyValidationService）のテスト

WFA検証 + 合格判定 + フィルタリングの各ロジックをテストします。
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.config.ga_config import (
    EvaluationConfig,
    ValidationConfig,
)
from app.services.auto_strategy.core.evaluation.evaluation_report import (
    EvaluationReport,
    ScenarioEvaluation,
)
from app.services.auto_strategy.services.strategy_validation_service import (
    StrategyValidationService,
)


def _make_report(
    pass_rate: float,
    primary_fitness: float | None = None,
    worst_case: float | None = None,
    scenarios: list[ScenarioEvaluation] | None = None,
    n_folds: int = 5,
) -> EvaluationReport:
    """テスト用の EvaluationReport を構築する。

    pass_rate は scenarios の passed フラグから算出されるため、
    指定した pass_rate に合うようにシナリオの passed を調整します。
    """
    if scenarios is None:
        passed_count = max(0, min(n_folds, round(pass_rate * n_folds)))
        scenarios = [
            ScenarioEvaluation(
                name=f"fold_{i}",
                fitness=(primary_fitness or 0.5,),
                passed=i < passed_count,
                performance_metrics={
                    "total_trades": 40,
                    "max_drawdown": 0.05,
                },
            )
            for i in range(n_folds)
        ]
    return EvaluationReport(
        mode="walk_forward",
        objectives=("total_return",),
        aggregated_fitness=(primary_fitness or 0.5,),
        scenarios=scenarios,
        aggregate_method="robust",
        metadata={"pass_rate": pass_rate, "scenario_count": len(scenarios)},
    )


@pytest.fixture
def mock_evaluator():
    """モック評価器"""
    evaluator = MagicMock()
    return evaluator


@pytest.fixture
def validation_config():
    """自動検証が有効な設定"""
    return GAConfig(
        validation_config=ValidationConfig(
            enabled=True,
            min_pass_rate=0.6,
            min_primary_fitness=None,
            min_trades=None,
            max_drawdown=None,
        ),
        evaluation_config=EvaluationConfig(
            enable_walk_forward=True,
            wfa_n_folds=5,
        ),
        objectives=["total_return"],
    )


class TestStrategyValidationService:
    """StrategyValidationService のテスト"""

    def test_validation_disabled_returns_result_unchanged(self, mock_evaluator):
        """検証が無効な場合は結果をそのまま返す"""
        config = GAConfig(validation_config=ValidationConfig(enabled=False))
        service = StrategyValidationService(mock_evaluator)

        result = {"best_strategy": "dummy", "all_strategies": []}
        filtered = service.validate_and_filter_result(result, config, {})

        assert filtered == result
        assert "validation_results" not in filtered

    def test_validation_enabled_runs_wfa(self, mock_evaluator, validation_config):
        """検証有効時は WFA 評価が実行される"""
        report = _make_report(pass_rate=0.8, primary_fitness=0.7)
        mock_report = MagicMock()
        mock_report.pass_rate = 0.8
        mock_report.primary_aggregated_fitness = 0.7
        mock_report.primary_worst_case_fitness = 0.4
        mock_report.mode = "walk_forward"
        mock_report.scenarios = report.scenarios

        service = StrategyValidationService(mock_evaluator)
        with patch.object(
            service._evaluation_strategy, "execute_report", return_value=mock_report
        ):
            best_strategy = MagicMock()
            best_strategy.id = "strategy-1"
            result = {
                "best_strategy": best_strategy,
                "best_fitness": 0.7,
                "all_strategies": [best_strategy],
                "fitness_scores": [0.7],
                "pareto_front": [],
                "evaluation_summaries": {},
            }

            filtered = service.validate_and_filter_result(result, validation_config, {})

        assert "validation_results" in filtered
        validation = filtered["validation_results"]["strategy-1"]
        assert validation["passed"] is True
        assert filtered["best_strategy"] is best_strategy

    def test_robustness_gate_keeps_wfa_passing_candidate(self, mock_evaluator):
        config = GAConfig(
            validation_config=ValidationConfig(enabled=True, min_pass_rate=0.6),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        config.robustness_config.enabled = True
        config.robustness_config.min_pass_rate = 1.0
        service = StrategyValidationService(mock_evaluator)
        best = MagicMock()
        best.id = "best"
        wfa_report = _make_report(pass_rate=1.0, primary_fitness=0.8)
        robustness_report = EvaluationReport.aggregate(
            mode="robustness",
            objectives=["total_return"],
            scenarios=[
                ScenarioEvaluation(name="base", fitness=(0.7,), passed=True),
                ScenarioEvaluation(name="stress", fitness=(0.6,), passed=True),
            ],
            metadata={"evaluation_layer": "robustness"},
        )
        mock_evaluator.evaluate_robustness_report.return_value = robustness_report

        with patch.object(
            service._evaluation_strategy, "execute_report", return_value=wfa_report
        ):
            filtered = service.validate_and_filter_result(
                {
                    "best_strategy": best,
                    "all_strategies": [best],
                    "fitness_scores": [0.8],
                    "pareto_front": [],
                },
                config,
                {},
            )

        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is True
        assert validation["robustness"]["passed"] is True
        assert filtered["best_strategy"] is best
        mock_evaluator.evaluate_robustness_report.assert_called_once()

    def test_robustness_gate_rejects_partial_failure(self, mock_evaluator):
        config = GAConfig(
            validation_config=ValidationConfig(enabled=True, min_pass_rate=0.6),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        config.robustness_config.enabled = True
        config.robustness_config.min_pass_rate = 1.0
        service = StrategyValidationService(mock_evaluator)
        best = MagicMock()
        best.id = "best"
        wfa_report = _make_report(pass_rate=1.0, primary_fitness=0.8)
        robustness_report = EvaluationReport.aggregate(
            mode="robustness",
            objectives=["total_return"],
            scenarios=[
                ScenarioEvaluation(name="base", fitness=(0.7,), passed=True),
            ],
            metadata={
                "evaluation_layer": "robustness",
                "evaluation_incomplete": True,
                "evaluation_failed": True,
            },
        )
        mock_evaluator.evaluate_robustness_report.return_value = robustness_report

        with patch.object(
            service._evaluation_strategy, "execute_report", return_value=wfa_report
        ):
            filtered = service.validate_and_filter_result(
                {
                    "best_strategy": best,
                    "all_strategies": [best],
                    "fitness_scores": [0.8],
                    "pareto_front": [],
                },
                config,
                {},
            )

        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is False
        assert validation["robustness"]["passed"] is False
        assert filtered["best_strategy"] is None
        assert any("Robustness gate" in reason for reason in validation["reasons"])

    def test_robustness_gate_rejects_execution_error(self, mock_evaluator):
        config = GAConfig(
            validation_config=ValidationConfig(enabled=True, min_pass_rate=0.6),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        config.robustness_config.enabled = True
        service = StrategyValidationService(mock_evaluator)
        best = MagicMock()
        best.id = "best"
        mock_evaluator.evaluate_robustness_report.side_effect = RuntimeError("boom")

        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            return_value=_make_report(pass_rate=1.0, primary_fitness=0.8),
        ):
            filtered = service.validate_and_filter_result(
                {
                    "best_strategy": best,
                    "all_strategies": [best],
                    "fitness_scores": [0.8],
                    "pareto_front": [],
                },
                config,
                {},
            )

        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is False
        assert validation["robustness"]["passed"] is False
        assert filtered["best_strategy"] is None

    def test_robustness_gate_driven_by_evaluation_plan(self, mock_evaluator):
        """評価計画のrobustness設定からゲートが有効化される"""
        config = GAConfig(
            validation_config=ValidationConfig(enabled=True, min_pass_rate=0.6),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
            evaluation_plan={
                "robustness": {
                    "enabled": True,
                    "scenarios": [
                        {"type": "symbol", "symbol": "ETH/USDT:USDT"},
                    ],
                    "policy": "gate_only",
                    "failure_policy": "fail_closed",
                }
            },
        )
        config.evaluation_plan.apply_to_ga_config(config)
        assert config.robustness_config.enabled is True
        assert config.robustness_config.validation_symbols == ["ETH/USDT:USDT"]

        service = StrategyValidationService(mock_evaluator)
        best = MagicMock()
        best.id = "best"
        wfa_report = _make_report(pass_rate=1.0, primary_fitness=0.8)
        robustness_report = EvaluationReport.aggregate(
            mode="robustness",
            objectives=["total_return"],
            scenarios=[
                ScenarioEvaluation(name="base", fitness=(0.7,), passed=True),
                ScenarioEvaluation(name="symbol_eth", fitness=(0.6,), passed=True),
            ],
            metadata={"evaluation_layer": "robustness"},
        )
        mock_evaluator.evaluate_robustness_report.return_value = robustness_report

        with patch.object(
            service._evaluation_strategy, "execute_report", return_value=wfa_report
        ):
            filtered = service.validate_and_filter_result(
                {
                    "best_strategy": best,
                    "all_strategies": [best],
                    "fitness_scores": [0.8],
                    "pareto_front": [],
                },
                config,
                {},
            )

        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is True
        assert validation["robustness"]["passed"] is True
        assert filtered["best_strategy"] is best
        mock_evaluator.evaluate_robustness_report.assert_called_once()

    def test_best_strategy_fails_but_candidate_passes(self, mock_evaluator):
        """最良戦略が不合格でも合格した候補があれば昇格する"""
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                min_pass_rate=0.6,
            ),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)

        best = MagicMock()
        best.id = "best"
        candidate = MagicMock()
        candidate.id = "candidate"

        mock_failing = _make_report(pass_rate=0.2, primary_fitness=0.1)

        mock_passing = _make_report(pass_rate=0.9, primary_fitness=0.8)

        def fake_execute_report(gene, base_config, ga_config):
            if gene.id == "best":
                return mock_failing
            return mock_passing

        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            side_effect=fake_execute_report,
        ):
            result = {
                "best_strategy": best,
                "best_fitness": 0.1,
                "all_strategies": [best, candidate],
                "fitness_scores": [0.1, 0.8],
                "pareto_front": [],
                "evaluation_summaries": {
                    "candidate": {"pass_rate": 0.9},
                },
            }
            filtered = service.validate_and_filter_result(result, config, {})

        # 最良戦略は候補に昇格する
        assert filtered["best_strategy"] is candidate
        assert filtered["best_fitness"] == 0.8
        assert filtered["best_evaluation_summary"] == {"pass_rate": 0.9}
        # all_strategies は合格した候補のみ
        assert filtered["all_strategies"] == [candidate]
        assert filtered["fitness_scores"] == [0.8]

    def test_all_strategies_fail(self, mock_evaluator):
        """全戦略が不合格なら best_strategy は None になる"""
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                min_pass_rate=0.9,
            ),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)

        best = MagicMock()
        best.id = "best"

        mock_failing = _make_report(pass_rate=0.2, primary_fitness=0.1)

        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            return_value=mock_failing,
        ):
            result = {
                "best_strategy": best,
                "best_fitness": 0.1,
                "all_strategies": [best],
                "fitness_scores": [0.1],
                "pareto_front": [],
                "evaluation_summaries": {},
            }
            filtered = service.validate_and_filter_result(result, config, {})

        assert filtered["best_strategy"] is None
        assert filtered["all_strategies"] == []
        assert filtered["pareto_front"] == []

    def test_min_trades_check(self, mock_evaluator):
        """min_trades 未満の戦略は不合格になる"""
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                min_pass_rate=0.0,
                min_trades=50,
            ),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)

        best = MagicMock()
        best.id = "best"

        low_trades_report = _make_report(
            pass_rate=1.0,
            scenarios=[
                ScenarioEvaluation(
                    name="fold_0",
                    fitness=(0.5,),
                    passed=True,
                    performance_metrics={"total_trades": 10, "max_drawdown": 0.05},
                )
            ],
            n_folds=1,
        )

        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            return_value=low_trades_report,
        ):
            result = {
                "best_strategy": best,
                "best_fitness": 0.5,
                "all_strategies": [best],
                "fitness_scores": [0.5],
                "pareto_front": [],
                "evaluation_summaries": {},
            }
            filtered = service.validate_and_filter_result(result, config, {})

        assert filtered["best_strategy"] is None
        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is False
        assert any("取引数" in reason for reason in validation["reasons"])

    def test_max_drawdown_check(self, mock_evaluator):
        """max_drawdown を超える戦略は不合格になる"""
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                min_pass_rate=0.0,
                max_drawdown=0.05,
            ),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)

        best = MagicMock()
        best.id = "best"

        high_dd_report = _make_report(
            pass_rate=1.0,
            scenarios=[
                ScenarioEvaluation(
                    name="fold_0",
                    fitness=(0.5,),
                    passed=True,
                    performance_metrics={"total_trades": 100, "max_drawdown": 0.3},
                )
            ],
            n_folds=1,
        )

        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            return_value=high_dd_report,
        ):
            result = {
                "best_strategy": best,
                "best_fitness": 0.5,
                "all_strategies": [best],
                "fitness_scores": [0.5],
                "pareto_front": [],
                "evaluation_summaries": {},
            }
            filtered = service.validate_and_filter_result(result, config, {})

        assert filtered["best_strategy"] is None
        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is False
        assert any("ドローダウン" in reason for reason in validation["reasons"])

    def test_missing_required_metrics_fail_closed(self, mock_evaluator):
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                min_pass_rate=0.0,
                min_trades=1,
                max_drawdown=0.5,
            ),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)
        best = MagicMock()
        best.id = "best"
        report = EvaluationReport(
            mode="walk_forward",
            objectives=("total_return",),
            aggregated_fitness=(0.8,),
            scenarios=[
                ScenarioEvaluation(
                    name="fold_0",
                    fitness=(0.8,),
                    passed=True,
                    performance_metrics={"total_trades": 10},
                ),
                ScenarioEvaluation(
                    name="fold_1",
                    fitness=(0.8,),
                    passed=True,
                    performance_metrics={"max_drawdown": 0.1},
                ),
            ],
        )

        with patch.object(
            service._evaluation_strategy, "execute_report", return_value=report
        ):
            filtered = service.validate_and_filter_result(
                {
                    "best_strategy": best,
                    "all_strategies": [best],
                    "fitness_scores": [0.8],
                    "pareto_front": [],
                },
                config,
                {},
            )

        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is False
        assert any("total_trades" in reason for reason in validation["reasons"])
        assert any("max_drawdown" in reason for reason in validation["reasons"])
        assert filtered["best_strategy"] is None

    def test_empty_report_fails_even_with_zero_pass_rate_threshold(
        self, mock_evaluator
    ):
        config = GAConfig(
            validation_config=ValidationConfig(enabled=True, min_pass_rate=0.0),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)
        best = MagicMock()
        best.id = "best"
        report = EvaluationReport(
            mode="walk_forward",
            objectives=("total_return",),
            aggregated_fitness=(),
            scenarios=[],
        )

        with patch.object(
            service._evaluation_strategy, "execute_report", return_value=report
        ):
            filtered = service.validate_and_filter_result(
                {
                    "best_strategy": best,
                    "all_strategies": [best],
                    "fitness_scores": [0.8],
                    "pareto_front": [],
                },
                config,
                {},
            )

        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is False
        assert any("シナリオがありません" in reason for reason in validation["reasons"])

    def test_incomplete_evaluation_fails_closed(self, mock_evaluator):
        config = GAConfig(
            validation_config=ValidationConfig(enabled=True, min_pass_rate=0.0),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)
        best = MagicMock()
        best.id = "best"
        report = _make_report(pass_rate=1.0, primary_fitness=0.8)
        report.metadata.update(
            {
                "evaluation_incomplete": True,
                "expected_fold_count": 5,
                "completed_fold_count": 4,
            }
        )

        with patch.object(
            service._evaluation_strategy, "execute_report", return_value=report
        ):
            filtered = service.validate_and_filter_result(
                {
                    "best_strategy": best,
                    "all_strategies": [best],
                    "fitness_scores": [0.8],
                    "pareto_front": [],
                },
                config,
                {},
            )

        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is False
        assert any("不完全" in reason for reason in validation["reasons"])
        assert (
            filtered["validation_report_summaries"]["best"]["metadata"][
                "evaluation_incomplete"
            ]
            is True
        )

    def test_evaluation_fallback_fails_closed(self, mock_evaluator):
        config = GAConfig(
            validation_config=ValidationConfig(enabled=True, min_pass_rate=0.0),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)
        best = MagicMock()
        best.id = "best"
        report = _make_report(pass_rate=1.0, primary_fitness=0.8)
        report.metadata.update(
            {"evaluation_fallback": True, "fallback_reason": "test failure"}
        )

        with patch.object(
            service._evaluation_strategy, "execute_report", return_value=report
        ):
            filtered = service.validate_and_filter_result(
                {
                    "best_strategy": best,
                    "all_strategies": [best],
                    "fitness_scores": [0.8],
                    "pareto_front": [],
                },
                config,
                {},
            )

        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is False
        assert any("フォールバック" in reason for reason in validation["reasons"])

    def test_min_primary_fitness_minimize_direction(self, mock_evaluator):
        """最小化目的（max_drawdown）では min_primary_fitness は上限として判定される"""
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                min_pass_rate=0.0,
                min_primary_fitness=0.1,  # 最大許容値（ドローダウン上限）
            ),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["max_drawdown"],
        )
        service = StrategyValidationService(mock_evaluator)

        best = MagicMock()
        best.id = "best"

        # max_drawdown 目的: fitness 0.3 は上限 0.1 を超える → 不合格
        over_limit_report = _make_report(
            pass_rate=1.0,
            primary_fitness=0.3,
        )
        over_limit_report.objectives = ("max_drawdown",)

        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            return_value=over_limit_report,
        ):
            result = {
                "best_strategy": best,
                "best_fitness": 0.3,
                "all_strategies": [best],
                "fitness_scores": [0.3],
                "pareto_front": [],
                "evaluation_summaries": {},
            }
            filtered = service.validate_and_filter_result(result, config, {})

        assert filtered["best_strategy"] is None
        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is False

        # max_drawdown 目的: fitness 0.05 は上限 0.1 以下 → 合格
        within_limit_report = _make_report(
            pass_rate=1.0,
            primary_fitness=0.05,
        )
        within_limit_report.objectives = ("max_drawdown",)

        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            return_value=within_limit_report,
        ):
            result2 = {
                "best_strategy": best,
                "best_fitness": 0.05,
                "all_strategies": [best],
                "fitness_scores": [0.05],
                "pareto_front": [],
                "evaluation_summaries": {},
            }
            filtered2 = service.validate_and_filter_result(result2, config, {})

        assert filtered2["best_strategy"] is best
        assert filtered2["validation_results"]["best"]["passed"] is True

    def test_pareto_only_promotion(self, mock_evaluator):
        """all_strategies が空でも合格したパレート戦略があれば昇格する"""
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                min_pass_rate=0.6,
            ),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)

        best = MagicMock()
        best.id = "best"
        pareto_pass = MagicMock()
        pareto_pass.id = "pareto-pass"

        def fake_execute_report(gene, base_config, ga_config):
            if gene.id == "best":
                return _make_report(pass_rate=0.2, primary_fitness=0.1)
            return _make_report(pass_rate=0.8, primary_fitness=0.7)

        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            side_effect=fake_execute_report,
        ):
            result = {
                "best_strategy": best,
                "best_fitness": 0.1,
                "all_strategies": [],  # 空
                "fitness_scores": [],
                "pareto_front": [
                    {"strategy": pareto_pass, "fitness_values": [0.7]},
                ],
                "evaluation_summaries": {
                    "pareto-pass": {"pass_rate": 0.8},
                },
            }
            filtered = service.validate_and_filter_result(result, config, {})

        assert filtered["best_strategy"] is pareto_pass
        assert filtered["best_evaluation_summary"] == {"pass_rate": 0.8}

    def test_validation_error_treated_as_fail(self, mock_evaluator):
        """検証エラー時は不合格として扱われる"""
        config = GAConfig(
            validation_config=ValidationConfig(enabled=True),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)

        best = MagicMock()
        best.id = "best"

        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            side_effect=RuntimeError("boom"),
        ):
            result = {
                "best_strategy": best,
                "best_fitness": 0.5,
                "all_strategies": [best],
                "fitness_scores": [0.5],
                "pareto_front": [],
                "evaluation_summaries": {},
            }
            filtered = service.validate_and_filter_result(result, config, {})

        validation = filtered["validation_results"]["best"]
        assert validation["passed"] is False
        assert validation["mode"] == "error"
        assert filtered["best_strategy"] is None

    def test_pareto_front_filtered(self, mock_evaluator):
        """パレートフロントも合格した戦略のみに絞られる"""
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                min_pass_rate=0.6,
            ),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)

        best = MagicMock()
        best.id = "best"
        pareto_pass = MagicMock()
        pareto_pass.id = "pareto-pass"
        pareto_fail = MagicMock()
        pareto_fail.id = "pareto-fail"

        def fake_execute_report(gene, base_config, ga_config):
            if gene.id == "pareto-fail":
                return _make_report(pass_rate=0.1, primary_fitness=0.1)
            return _make_report(pass_rate=0.8, primary_fitness=0.7)

        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            side_effect=fake_execute_report,
        ):
            result = {
                "best_strategy": best,
                "best_fitness": 0.7,
                "all_strategies": [best],
                "fitness_scores": [0.7],
                "pareto_front": [
                    {"strategy": pareto_pass, "fitness_values": [0.7]},
                    {"strategy": pareto_fail, "fitness_values": [0.1]},
                ],
                "evaluation_summaries": {},
            }
            filtered = service.validate_and_filter_result(result, config, {})

        assert len(filtered["pareto_front"]) == 1
        assert filtered["pareto_front"][0]["strategy"] is pareto_pass

    def test_duplicate_candidate_key_validated_once(self, mock_evaluator):
        """同一キーの候補が重複しても検証は1回だけ実行される"""
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                min_pass_rate=0.6,
            ),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)

        best = MagicMock()
        best.id = "best"
        # 同一キーの戦略が all_strategies に重複して含まれるケース
        dup1 = MagicMock()
        dup1.id = "dup"
        dup2 = MagicMock()
        dup2.id = "dup"

        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            return_value=_make_report(pass_rate=0.9, primary_fitness=0.8),
        ) as mock_execute:
            result = {
                "best_strategy": best,
                "best_fitness": 0.8,
                "all_strategies": [dup1, dup2],
                "fitness_scores": [0.8, 0.8],
                "pareto_front": [],
                "evaluation_summaries": {},
            }
            filtered = service.validate_and_filter_result(result, config, {})

        # best + dup の2キーのみ検証される（dup は重複スキップ）
        assert mock_execute.call_count == 2
        assert set(filtered["validation_results"].keys()) == {"best", "dup"}

    def test_min_primary_fitness_maximize_direction(self, mock_evaluator):
        """最大化目的（total_return）では min_primary_fitness 未満は不合格になる"""
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                min_pass_rate=0.0,
                min_primary_fitness=0.5,
            ),
            evaluation_config=EvaluationConfig(enable_walk_forward=True),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)

        best = MagicMock()
        best.id = "best"

        # total_return 目的: fitness 0.3 は下限 0.5 未満 → 不合格
        below_report = _make_report(pass_rate=1.0, primary_fitness=0.3)
        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            return_value=below_report,
        ):
            result = {
                "best_strategy": best,
                "best_fitness": 0.3,
                "all_strategies": [best],
                "fitness_scores": [0.3],
                "pareto_front": [],
                "evaluation_summaries": {},
            }
            filtered = service.validate_and_filter_result(result, config, {})

        assert filtered["best_strategy"] is None
        assert filtered["validation_results"]["best"]["passed"] is False
        assert any(
            "集約フィットネス" in reason
            for reason in filtered["validation_results"]["best"]["reasons"]
        )

        # total_return 目的: fitness 0.7 は下限 0.5 以上 → 合格
        within_report = _make_report(pass_rate=1.0, primary_fitness=0.7)
        with patch.object(
            service._evaluation_strategy,
            "execute_report",
            return_value=within_report,
        ):
            result2 = {
                "best_strategy": best,
                "best_fitness": 0.7,
                "all_strategies": [best],
                "fitness_scores": [0.7],
                "pareto_front": [],
                "evaluation_summaries": {},
            }
            filtered2 = service.validate_and_filter_result(result2, config, {})

        assert filtered2["best_strategy"] is best
        assert filtered2["validation_results"]["best"]["passed"] is True

    def test_get_strategy_key_delegates_to_common_helper(self):
        """``_get_strategy_key`` が共通ヘルパーへ委譲することの確認

        キー生成ロジック自体は ``GeneticUtils.get_strategy_result_key`` のテスト
        （test_genetic_utils.py）で検証済みのため、ここでは委譲のみを確認します。
        """
        strategy = object()

        with patch(
            "app.services.auto_strategy.services.strategy_validation_service.GeneticUtils.get_strategy_result_key",
            return_value="mocked-key",
        ) as mock_helper:
            key = StrategyValidationService._get_strategy_key(strategy)
            assert key == "mocked-key"

        mock_helper.assert_called_once_with(strategy)


class TestValidationGaConfigBuilding:
    """_build_validation_ga_config の設定テスト"""

    def test_build_validation_ga_config_disables_early_termination(
        self, mock_evaluator
    ):
        """WFA検証用設定では早期終了が無効化される"""
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                wfa_n_folds=3,
                wfa_train_ratio=0.7,
            ),
            evaluation_config=EvaluationConfig(
                enable_walk_forward=False,
                oos_split_ratio=0.25,
            ),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)

        validation_config = service._build_validation_ga_config(config)

        eval_cfg = validation_config.evaluation_config
        # WFAが有効化され、OOS/PurgedKFold は無効化される
        assert eval_cfg.enable_walk_forward is True
        assert eval_cfg.wfa_n_folds == 3
        assert eval_cfg.oos_split_ratio == 0.0
        assert validation_config.enable_purged_kfold is False
        # 最終品質ゲートは完全評価で行う（早期終了を無効化）
        assert eval_cfg.early_termination_settings.enabled is False

    def test_build_validation_ga_config_preserves_other_settings(self, mock_evaluator):
        """検証用設定で他の評価設定は維持される"""
        config = GAConfig(
            validation_config=ValidationConfig(enabled=True, wfa_n_folds=5),
            evaluation_config=EvaluationConfig(
                enable_walk_forward=False,
                early_termination_settings={
                    "enabled": True,
                    "min_trades": 30,
                },
            ),
            objectives=["total_return"],
        )
        service = StrategyValidationService(mock_evaluator)

        validation_config = service._build_validation_ga_config(config)

        assert validation_config.objectives == ["total_return"]
        assert validation_config.evaluation_config.wfa_n_folds == 5


class TestValidationConfig:
    """ValidationConfig の設定テスト"""

    def test_defaults(self):
        """デフォルト値のテスト"""
        config = GAConfig()
        vc = config.validation_config
        assert vc.enabled is True
        assert vc.min_pass_rate == 0.5
        assert vc.min_primary_fitness is None
        assert vc.min_trades is None
        assert vc.max_drawdown is None
        assert vc.wfa_n_folds == 5
        assert vc.wfa_train_ratio == 0.7
        assert vc.wfa_anchored is False
        assert vc.validate_candidates is True
        assert vc.max_candidates == 5

    def test_from_dict(self):
        """辞書からの復元テスト"""
        data = {
            "validation_config": {
                "enabled": True,
                "min_pass_rate": 0.7,
                "min_trades": 20,
                "max_drawdown": 0.1,
                "wfa_n_folds": 4,
                "validate_candidates": False,
            }
        }
        config = GAConfig.from_dict(data)
        vc = config.validation_config
        assert vc.enabled is True
        assert vc.min_pass_rate == 0.7
        assert vc.min_trades == 20
        assert vc.max_drawdown == 0.1
        assert vc.wfa_n_folds == 4
        assert vc.validate_candidates is False

    def test_to_dict_round_trip(self):
        """to_dict → from_dict のラウンドトリップテスト"""
        config = GAConfig(
            validation_config=ValidationConfig(
                enabled=True,
                min_pass_rate=0.65,
                min_primary_fitness=0.2,
                max_drawdown=0.15,
            )
        )
        restored = GAConfig.from_dict(config.to_dict())
        vc = restored.validation_config
        assert vc.enabled is True
        assert vc.min_pass_rate == 0.65
        assert vc.min_primary_fitness == 0.2
        assert vc.max_drawdown == 0.15

    def test_exported_from_package(self):
        """パッケージから ValidationConfig がエクスポートされる"""
        from app.services.auto_strategy.config import ValidationConfig as V2

        assert V2 is ValidationConfig
