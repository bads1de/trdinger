from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from app.services.auto_strategy.core.evaluation.evaluation_strategies import (
    EvaluationStrategy,
)
from app.services.auto_strategy.core.evaluation.evaluation_report import (
    ScenarioEvaluation,
)


class TestEvaluationStrategy:
    def setup_method(self):
        self.evaluator = Mock()
        self.evaluator._perform_single_evaluation_report = Mock(
            side_effect=lambda _gene, _backtest_config, _config, **kwargs: ScenarioEvaluation(
                name=kwargs.get("scenario_name", "single"),
                fitness=(0.0,),
                passed=True,
                metadata=(kwargs.get("metadata") or {}).copy(),
            )
        )
        self.strategy = EvaluationStrategy(self.evaluator)

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

        result = self.strategy.execute(
            object(),
            {
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-02-01 00:00:00",
            },
            config,
        )

        assert result == (0.42,)
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

        result = self.strategy.execute(
            object(),
            {
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-01-11 00:00:00",
            },
            config,
        )

        # robust aggregation: median(1.0, 0.5) * 0.7 + min(1.0, 0.5) * 0.3 = 0.75 * 0.7 + 0.5 * 0.3 = 0.675
        assert result[0] == pytest.approx(0.675)

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

    def test_execute_report_falls_back_to_legacy_single_evaluation_method(self):
        evaluator = Mock()
        evaluator._perform_single_evaluation = Mock(return_value=(0.33,))
        strategy = EvaluationStrategy(evaluator)
        gene = object()

        config = SimpleNamespace(
            enable_purged_kfold=False,
            evaluation_config=SimpleNamespace(
                enable_walk_forward=False,
                oos_split_ratio=0.0,
            ),
            objectives=["weighted_score"],
            fitness_constraints={},
        )

        backtest_config = {
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-01-11 00:00:00",
        }

        report = strategy.execute_report(gene, backtest_config, config)

        assert report.mode == "single"
        assert report.aggregated_fitness == (0.33,)
        evaluator._perform_single_evaluation.assert_called_once_with(
            gene,
            backtest_config,
            config,
        )
        assert [scenario.name for scenario in report.scenarios] == ["single"]

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
