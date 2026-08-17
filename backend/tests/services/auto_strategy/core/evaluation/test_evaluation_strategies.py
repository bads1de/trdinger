from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from app.services.auto_strategy.core.evaluation.evaluation_report import (
    ScenarioEvaluation,
)
from app.services.auto_strategy.core.evaluation.evaluation_strategies import (
    EvaluationStrategy,
    resolve_evaluation_mode,
)


def _config(**eval_kwargs: object) -> SimpleNamespace:
    """evaluation_config と enable_purged_kfold（GAConfig トップレベル）を持つテスト用設定を生成する。"""
    purged = bool(eval_kwargs.pop("enable_purged_kfold", False))
    defaults = {
        "evaluation_mode": "auto",
        "enable_walk_forward": False,
        "oos_split_ratio": 0.0,
    }
    defaults.update(eval_kwargs)
    return SimpleNamespace(
        enable_purged_kfold=purged,
        evaluation_config=SimpleNamespace(**defaults),
    )


class TestResolveEvaluationMode:
    """evaluation_mode 解決ロジックのテスト。"""

    def test_explicit_mode_wins_over_flags(self):
        # 明示的に walk_forward を指定しても、enable_purged_kfold=True が競合しても WFA 優先
        cfg = _config(
            evaluation_mode="walk_forward",
            enable_purged_kfold=True,
            enable_walk_forward=False,
            oos_split_ratio=0.25,
        )
        assert resolve_evaluation_mode(cfg) == "walk_forward"

    def test_explicit_oos_mode(self):
        cfg = _config(evaluation_mode="oos", oos_split_ratio=0.0)
        assert resolve_evaluation_mode(cfg) == "oos"

    def test_explicit_single_mode(self):
        cfg = _config(evaluation_mode="single", oos_split_ratio=0.25)
        assert resolve_evaluation_mode(cfg) == "single"

    def test_auto_derives_purged_kfold_from_flag(self):
        cfg = _config(enable_purged_kfold=True, oos_split_ratio=0.25)
        assert resolve_evaluation_mode(cfg) == "purged_kfold"

    def test_auto_derives_walk_forward_from_flag(self):
        cfg = _config(enable_walk_forward=True, oos_split_ratio=0.25)
        assert resolve_evaluation_mode(cfg) == "walk_forward"

    def test_auto_derives_oos_from_ratio(self):
        cfg = _config(oos_split_ratio=0.25)
        assert resolve_evaluation_mode(cfg) == "oos"

    def test_auto_derives_single_when_all_off(self):
        cfg = _config()
        assert resolve_evaluation_mode(cfg) == "single"

    def test_unknown_explicit_mode_falls_back_to_flags(self):
        cfg = _config(evaluation_mode="bogus", oos_split_ratio=0.3)
        assert resolve_evaluation_mode(cfg) == "oos"


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
        # 明示モード（walk_forward）は enable_purged_kfold フラグより優先される
        cfg = _config(
            evaluation_mode="walk_forward",
            enable_purged_kfold=True,
        )
        self.strategy._evaluate_with_walk_forward_report = Mock(
            return_value=Mock(aggregated_fitness=(0.7,))
        )
        self.strategy._evaluate_with_purged_kfold_report = Mock()
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
        self.strategy._evaluate_with_purged_kfold_report.assert_not_called()

    def test_execute_prefers_purged_kfold_when_enabled(self):
        config = SimpleNamespace(
            enable_purged_kfold=True,
            evaluation_config=SimpleNamespace(
                enable_walk_forward=False,
                oos_split_ratio=0.0,
            ),
        )
        self.strategy._evaluate_with_purged_kfold_report = Mock(
            return_value=Mock(aggregated_fitness=(0.42,))
        )

        report = self.strategy.execute_report(
            object(),
            {
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-02-01 00:00:00",
            },
            config,
        )

        assert report.aggregated_fitness == (0.42,)
        self.strategy._evaluate_with_purged_kfold_report.assert_called_once()
        self.evaluator._perform_single_evaluation_report.assert_not_called()

    def test_purged_kfold_averages_test_fold_results(self):
        def side_effect(
            _gene,
            backtest_config,
            _config,
            *,
            scenario_name="single",
            metadata=None,
        ):
            start_date = str(backtest_config["start_date"])
            if start_date == "2024-01-01 00:00:00":
                fitness = (1.0,)
            elif start_date == "2024-01-06 00:00:00":
                fitness = (0.5,)
            else:
                fitness = (0.0,)
            return ScenarioEvaluation(
                name=scenario_name,
                fitness=fitness,
                passed=True,
                metadata=(metadata or {}).copy(),
            )

        self.evaluator._perform_single_evaluation_report.side_effect = side_effect

        config = SimpleNamespace(
            enable_purged_kfold=True,
            purged_kfold_splits=2,
            purged_kfold_embargo=0.0,
            evaluation_config=SimpleNamespace(
                enable_walk_forward=False,
                oos_split_ratio=0.0,
            ),
            objectives=["weighted_score"],
        )

        report = self.strategy.execute_report(
            object(),
            {
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-01-11 00:00:00",
            },
            config,
        )

        # robust aggregation: median(1.0, 0.5) * 0.7 + min(1.0, 0.5) * 0.3 = 0.75 * 0.7 + 0.5 * 0.3 = 0.675
        assert report.aggregated_fitness[0] == pytest.approx(0.675)

    def test_execute_report_returns_scenarios_for_oos(self):
        self.evaluator._perform_single_evaluation_report.side_effect = [
            ScenarioEvaluation(name="is", fitness=(1.0,), passed=True, metadata={}),
            ScenarioEvaluation(name="oos", fitness=(0.4,), passed=True, metadata={}),
        ]

        config = SimpleNamespace(
            enable_purged_kfold=False,
            evaluation_config=SimpleNamespace(
                enable_walk_forward=False,
                oos_split_ratio=0.2,
                oos_fitness_weight=0.75,
            ),
            objectives=["weighted_score"],
            fitness_constraints={},
        )

        report = self.strategy.execute_report(
            object(),
            {
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-01-11 00:00:00",
            },
            config,
        )

        assert report.mode == "oos"
        assert len(report.scenarios) == 2
        assert [scenario.name for scenario in report.scenarios] == ["is", "oos"]
        assert report.aggregated_fitness == (0.55,)

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

    def test_execute_report_falls_back_to_single_when_date_range_missing(self):
        single_report = Mock()
        self.strategy._evaluate_single_report = Mock(return_value=single_report)

        config = SimpleNamespace(
            enable_purged_kfold=False,
            evaluation_config=SimpleNamespace(
                enable_walk_forward=False,
                oos_split_ratio=0.2,
                oos_fitness_weight=0.75,
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
            enable_purged_kfold=False,
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

    def test_precompute_purged_kfold_configs_does_not_add_internal_keys(self):
        configs = self.strategy._precompute_purged_kfold_configs(
            start_date=pd.Timestamp("2024-01-01 00:00:00"),
            end_date=pd.Timestamp("2024-01-11 00:00:00"),
            n_splits=2,
            embargo_pct=0.1,
            base_backtest_config={"symbol": "BTCUSDT", "timeframe": "1h"},
        )

        assert len(configs) == 2
        for _, config in configs:
            assert "_embargo_seconds" not in config
            assert config["symbol"] == "BTCUSDT"
            assert config["timeframe"] == "1h"

    def test_precompute_purged_kfold_configs_applies_embargo_gap(self):
        configs = self.strategy._precompute_purged_kfold_configs(
            start_date=pd.Timestamp("2024-01-01 00:00:00"),
            end_date=pd.Timestamp("2024-01-11 00:00:00"),
            n_splits=2,
            embargo_pct=0.1,
            base_backtest_config={"symbol": "BTCUSDT", "timeframe": "1h"},
        )

        assert len(configs) == 2
        assert configs[0][1]["end_date"] == "2024-01-05 12:00:00"
        assert configs[1][1]["start_date"] == "2024-01-06 12:00:00"

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
            enable_purged_kfold=False,
            evaluation_config=SimpleNamespace(
                enable_walk_forward=False,
                oos_split_ratio=0.0,
            ),
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
            enable_purged_kfold=False,
            evaluation_config=SimpleNamespace(
                enable_walk_forward=False,
                oos_split_ratio=0.0,
            ),
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
            enable_purged_kfold=False,
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

    def test_purged_kfold_report_falls_back_when_all_fold_evaluations_fail(self):
        self.strategy._evaluate_single_report = Mock(
            return_value=Mock(mode="single", aggregated_fitness=(0.18,))
        )
        self.strategy._precompute_purged_kfold_configs = Mock(
            return_value=[
                (
                    0,
                    {
                        "start_date": "2024-01-01 00:00:00",
                        "end_date": "2024-01-11 00:00:00",
                    },
                )
            ]
        )
        self.strategy._evaluate_scenario = Mock(side_effect=RuntimeError("fold failed"))

        config = SimpleNamespace(
            enable_purged_kfold=True,
            purged_kfold_splits=3,
            purged_kfold_embargo=0.0,
            evaluation_config=SimpleNamespace(
                enable_walk_forward=False,
                oos_split_ratio=0.0,
            ),
            objectives=["weighted_score"],
            fitness_constraints={},
        )

        report = self.strategy._evaluate_with_purged_kfold_report(
            object(),
            {
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-02-01 00:00:00",
            },
            config,
        )

        assert report.mode == "single"
        assert report.aggregated_fitness == (0.18,)
        self.strategy._evaluate_single_report.assert_called_once()
