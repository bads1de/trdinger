import importlib.util
import os
import sys
from datetime import datetime

from database.models import BacktestResult, GeneratedStrategy


current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
sys.path.insert(0, backend_dir)

generated_strategy_service_path = os.path.join(
    backend_dir,
    "app",
    "services",
    "auto_strategy",
    "services",
    "generated_strategy_service.py",
)
spec = importlib.util.spec_from_file_location(
    "generated_strategy_service",
    generated_strategy_service_path,
)
generated_strategy_service_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generated_strategy_service_module)
GeneratedStrategyService = generated_strategy_service_module.GeneratedStrategyService


class _DummyGeneratedStrategyRepo:
    def __init__(self, strategies):
        self._strategies = strategies

    def get_filtered_and_sorted_strategies(
        self,
        limit=50,
        offset=0,
        risk_level=None,
        experiment_id=None,
        min_fitness=None,
        sort_by="fitness_score",
        sort_order="desc",
    ):
        return len(self._strategies), self._strategies[offset : offset + limit]


def test_get_strategies_exposes_evaluation_summary():
    service = GeneratedStrategyService(db=object())

    strategy = GeneratedStrategy(
        id=7,
        experiment_id=11,
        generation=12,
        fitness_score=1.25,
        gene_data={
            "indicators": [{"type": "rsi", "enabled": True}],
            "risk_management": {},
            "long_entry_conditions": {},
            "short_entry_conditions": {},
            "metadata": {
                "evaluation_summary": {
                    "mode": "robustness",
                    "pass_rate": 0.75,
                    "selection_rank": 0,
                    "scenarios": [
                        {"name": "base", "passed": True, "fitness": [0.9]},
                        {
                            "name": "commission_x1.5",
                            "passed": False,
                            "fitness": [0.5],
                        },
                    ],
                }
            },
        },
        created_at=datetime(2024, 1, 1, 12, 0, 0),
    )
    strategy.backtest_result = BacktestResult(
        id=1,
        strategy_name="Test Strategy",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 2, 1),
        initial_capital=10000.0,
        config_json={},
        equity_curve=[],
        trade_history=[],
        performance_metrics={
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
            "win_rate": 0.61,
        },
    )

    service.generated_strategy_repo = _DummyGeneratedStrategyRepo([strategy])

    result = service.get_strategies(limit=10, offset=0)

    display = result["strategies"][0]
    assert display["evaluation_summary"]["mode"] == "robustness"
    assert display["evaluation_summary"]["pass_rate"] == 0.75
    assert display["robustness_pass_rate"] == 0.75
    assert display["selection_rank"] == 0
    assert display["evaluation_summary"]["scenarios"][1]["name"] == "commission_x1.5"
