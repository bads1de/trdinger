"""EvaluationStrategy のレジームウィンドウ関連テスト。"""

from types import SimpleNamespace

import pytest

from app.services.auto_strategy.core.evaluation.evaluation_report import (
    ScenarioEvaluation,
)
from app.services.auto_strategy.core.evaluation.evaluation_strategies import (
    EvaluationStrategy,
)


class _StubEvaluator:
    def __init__(self):
        self.calls = []

    def _perform_single_evaluation_report(
        self,
        _gene,
        backtest_config,
        _config,
        *,
        scenario_name="single",
        metadata=None,
    ):
        self.calls.append(
            (
                backtest_config["start_date"],
                backtest_config["end_date"],
                backtest_config["symbol"],
            )
        )
        metadata = (metadata or {}).copy()
        if str(backtest_config["start_date"]).startswith("2024-07-01"):
            return ScenarioEvaluation(
                name=scenario_name,
                fitness=(0.3,),
                passed=True,
                metadata=metadata,
            )
        return ScenarioEvaluation(
            name=scenario_name,
            fitness=(0.8,),
            passed=True,
            metadata=metadata,
        )


def test_execute_robustness_report_adds_regime_windows():
    evaluator = _StubEvaluator()
    strategy = EvaluationStrategy(evaluator)
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
            validation_symbols=[],
            stress_slippage=[],
            stress_commission_multipliers=[],
            regime_windows=[
                {
                    "name": " bear ",
                    "start_date": " 2024-07-01 00:00:00 ",
                    "end_date": " 2024-08-01 00:00:00 ",
                }
            ],
            aggregate_method="robust",
        ),
    )

    report = strategy.execute_robustness_report(
        object(),
        {
            "symbol": "BTCUSDT",
            "commission_rate": 0.001,
            "slippage": 0.001,
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-02-01 00:00:00",
        },
        config,
    )

    assert [scenario.name for scenario in report.scenarios] == ["base", "regime_bear"]
    assert report.scenarios[1].metadata["regime_name"] == "bear"
    assert report.scenarios[1].metadata["start_date"] == "2024-07-01 00:00:00"
    assert report.aggregated_fitness[0] == pytest.approx(0.475)
    assert evaluator.calls == [
        ("2024-01-01 00:00:00", "2024-02-01 00:00:00", "BTCUSDT"),
        ("2024-07-01 00:00:00", "2024-08-01 00:00:00", "BTCUSDT"),
    ]
