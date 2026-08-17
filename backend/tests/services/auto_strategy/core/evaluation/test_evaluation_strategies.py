from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services.auto_strategy.core.evaluation.evaluation_report import (
    ScenarioEvaluation,
)
from app.services.auto_strategy.core.evaluation.evaluation_strategies import (
    EvaluationStrategy,
    resolve_evaluation_mode,
)


def _config(**eval_kwargs: object) -> SimpleNamespace:
    """evaluation_config を持つテスト用設定を生成する。"""
    defaults = {
        "evaluation_mode": "auto",
        "enable_walk_forward": False,
    }
    defaults.update(eval_kwargs)
    return SimpleNamespace(evaluation_config=SimpleNamespace(**defaults))


class TestResolveEvaluationMode:
    """evaluation_mode 解決ロジックのテスト。"""

    def test_explicit_mode_wins_over_flags(self):
        cfg = _config(
            evaluation_mode="walk_forward",
            enable_walk_forward=False,
        )
        assert resolve_evaluation_mode(cfg) == "walk_forward"

    def test_explicit_single_mode(self):
        cfg = _config(evaluation_mode="single")
        assert resolve_evaluation_mode(cfg) == "single"

    def test_auto_derives_walk_forward_from_flag(self):
        cfg = _config(enable_walk_forward=True)
        assert resolve_evaluation_mode(cfg) == "walk_forward"

    def test_auto_derives_single_when_all_off(self):
        cfg = _config()
        assert resolve_evaluation_mode(cfg) == "single"

    def test_unknown_explicit_mode_falls_back_to_flags(self):
        cfg = _config(evaluation_mode="bogus", enable_walk_forward=True)
        assert resolve_evaluation_mode(cfg) == "walk_forward"


class TestEvaluationStrategy:
    def setup_method(self):
        self.evaluator = Mock()
        self.evaluator._perform_single_evaluation_report = Mock(
            side_effect=lambda _gene, _backtest_config, _config, **kwargs: (
                ScenarioEvaluation(
                    name=kwargs.get("scenario_name", "single"),
                    fitness=(0.0,),
                    passed=True,
                    metadata=(kwargs.get("metadata") or {}).copy(),
                )
            )
        )
        self.strategy = EvaluationStrategy(self.evaluator)

    def test_execute_report_routes_on_explicit_mode(self):
        cfg = _config(evaluation_mode="walk_forward")
        self.strategy._evaluate_with_walk_forward_report = Mock(
            return_value=Mock(aggregated_fitness=(0.7,))
        )
        report = self.strategy.execute_report(
            object(),
            {
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-02-01 00:00:00",
            },
            cfg,
        )
        assert report.aggregated_fitness == (0.7,)
        self.strategy._evaluate_with_walk_forward_report.assert_called_once()

    def test_execute_report_falls_back_to_single_by_default(self):
        single_report = Mock()
        self.strategy._evaluate_single_report = Mock(return_value=single_report)

        cfg = _config(
            enable_walk_forward=False,
        )

        report = self.strategy.execute_report(
            object(),
            {
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-01-11 00:00:00",
            },
            cfg,
        )

        assert report is single_report
        self.strategy._evaluate_single_report.assert_called_once()

    def test_execute_ga_report_uses_is_only_regardless_of_oos_weight(self):
        def side_effect(
            _gene,
            _backtest_config,
            _config,
            *,
            scenario_name="single",
            metadata=None,
        ):
            return ScenarioEvaluation(
                name=scenario_name,
                fitness=(0.8,),
                passed=True,
                metadata=(metadata or {}).copy(),
            )

        self.evaluator._perform_single_evaluation_report.side_effect = side_effect
        base_config = {
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-01-11 00:00:00",
            "timeframe": "1h",
        }

        config = SimpleNamespace(
            evaluation_config=SimpleNamespace(
                oos_split_ratio=0.2,
                oos_fitness_weight=0.1,
            ),
            objectives=["weighted_score"],
        )
        first = self.strategy.execute_ga_report(object(), base_config, config)
        config.evaluation_config.oos_fitness_weight = 0.9
        second = self.strategy.execute_ga_report(object(), base_config, config)

        assert first.mode == "in_sample"
        assert second.mode == "in_sample"
        assert first.aggregated_fitness == (0.8,)
        assert second.aggregated_fitness == (0.8,)
        assert [scenario.name for scenario in first.scenarios] == ["is"]
        assert first.metadata["holdout_reserved"] is True
        assert self.evaluator._perform_single_evaluation_report.call_count == 2

    def test_resolve_backtest_date_range_rejects_missing_or_invalid_ranges(self):
        assert self.strategy._resolve_backtest_date_range({}) is None
        assert (
            self.strategy._resolve_backtest_date_range(
                {
                    "start_date": "2024-01-02 00:00:00",
                    "end_date": "2024-01-01 00:00:00",
                }
            )
            is None
        )

    def test_walk_forward_report_falls_back_when_date_range_missing(self):
        single_report = Mock()
        self.strategy._evaluate_single_report = Mock(return_value=single_report)

        config = SimpleNamespace(
            evaluation_config=SimpleNamespace(
                enable_walk_forward=True,
                wfa_n_folds=3,
                wfa_train_ratio=0.8,
                wfa_anchored=False,
            ),
            objectives=["weighted_score"],
            fitness_constraints={},
        )

        report = self.strategy.execute_report(
            object(),
            {
                "end_date": "2024-01-11 00:00:00",
            },
            config,
        )

        assert report is single_report
        self.strategy._evaluate_single_report.assert_called_once()

    def test_execute_report_uses_nested_wfa_config(self):
        single_report = Mock()
        self.strategy._evaluate_single_report = Mock(return_value=single_report)
        self.strategy._precompute_fold_configs = Mock(return_value=[])

        config = SimpleNamespace(
            evaluation_config=SimpleNamespace(
                enable_walk_forward=True,
                wfa_n_folds=3,
                wfa_train_ratio=0.8,
                wfa_anchored=True,
            ),
            objectives=["weighted_score"],
            fitness_constraints={},
        )

        report = self.strategy._evaluate_with_walk_forward_report(
            object(),
            {
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-01-11 00:00:00",
            },
            config,
        )

        called_args = self.strategy._precompute_fold_configs.call_args.args
        assert called_args[2] == 3
        assert called_args[3] == 0.8
        assert called_args[4] is True
        assert report is single_report

    def test_execute_robustness_report_adds_symbol_and_cost_scenarios(self):
        def side_effect(
            _gene,
            backtest_config,
            _config,
            *,
            scenario_name="single",
            metadata=None,
        ):
            if backtest_config["symbol"] == "ETHUSDT":
                fitness = (0.7,)
            elif backtest_config.get("slippage") == 0.0015:
                fitness = (0.6,)
            elif backtest_config.get("commission_rate") == 0.0015:
                fitness = (0.65,)
            else:
                fitness = (0.8,)
            return ScenarioEvaluation(
                name=scenario_name,
                fitness=fitness,
                passed=True,
                metadata=(metadata or {}).copy(),
            )

        self.evaluator._perform_single_evaluation_report.side_effect = side_effect

        config = SimpleNamespace(
            evaluation_config=SimpleNamespace(enable_walk_forward=False),
            objectives=["weighted_score"],
            fitness_constraints={},
            two_stage_selection_config=SimpleNamespace(min_pass_rate=1.0),
            robustness_config=SimpleNamespace(
                validation_symbols=["ETHUSDT"],
                stress_slippage=[0.0005],
                stress_commission_multipliers=[1.5],
                aggregate_method="robust",
            ),
        )

        report = self.strategy.execute_robustness_report(
            object(),
            {
                "symbol": "BTCUSDT",
                "commission_rate": 0.001,
                "slippage": 0.001,
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-01-11 00:00:00",
            },
            config,
        )

        assert report.mode == "robustness"
        assert [scenario.name for scenario in report.scenarios] == [
            "base",
            "symbol_ETHUSDT",
            "slippage_0.0015",
            "commission_x1.5",
        ]
        assert report.pass_rate == 1.0
        assert report.aggregated_fitness[0] == pytest.approx(0.6525)

    def test_execute_robustness_report_falls_back_when_all_scenarios_fail(self):
        self.strategy._evaluate_robustness_scenario_report = Mock(
            side_effect=RuntimeError("scenario failed")
        )

        config = SimpleNamespace(
            evaluation_config=SimpleNamespace(enable_walk_forward=False),
            objectives=["weighted_score"],
            fitness_constraints={},
            two_stage_selection_config=SimpleNamespace(min_pass_rate=0.5),
            robustness_config=SimpleNamespace(
                validation_symbols=["ETHUSDT"],
                stress_slippage=[],
                stress_commission_multipliers=[],
                regime_windows=[],
                aggregate_method="robust",
            ),
        )

        report = self.strategy.execute_robustness_report(
            object(),
            {
                "symbol": "BTCUSDT",
                "commission_rate": 0.001,
                "slippage": 0.001,
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-01-11 00:00:00",
            },
            config,
        )

        assert report.mode == "robustness"
        assert report.scenarios == []
        assert report.metadata["evaluation_failed"] is True
        assert report.metadata["failed_scenario_count"] == 2

    def test_walk_forward_report_falls_back_when_all_fold_evaluations_fail(self):
        self.strategy._evaluate_single_report = Mock(
            return_value=Mock(mode="single", aggregated_fitness=(0.15,))
        )
        self.strategy._precompute_fold_configs = Mock(
            return_value=[
                (
                    1,
                    {
                        "start_date": "2024-01-01 00:00:00",
                        "end_date": "2024-01-11 00:00:00",
                    },
                )
            ]
        )
        self.strategy._evaluate_scenario = Mock(side_effect=RuntimeError("fold failed"))

        config = SimpleNamespace(
            evaluation_config=SimpleNamespace(
                enable_walk_forward=True,
                wfa_n_folds=3,
                wfa_train_ratio=0.8,
                wfa_anchored=False,
            ),
            objectives=["weighted_score"],
            fitness_constraints={},
        )

        report = self.strategy._evaluate_with_walk_forward_report(
            object(),
            {
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-02-01 00:00:00",
            },
            config,
        )

        assert report.mode == "single"
        assert report.aggregated_fitness == (0.15,)
        self.strategy._evaluate_single_report.assert_called_once()
