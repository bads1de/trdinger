"""Hybrid evaluator のスモークテスト。"""

from unittest.mock import Mock, patch

import pandas as pd

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.genes.conditions import Condition
from app.services.auto_strategy.genes.indicator import IndicatorGene
from app.services.auto_strategy.genes.strategy import StrategyGene


def _sample_gene() -> StrategyGene:
    indicator = IndicatorGene(type="SMA", parameters={"period": 20})
    condition = Condition(left_operand="close", operator=">", right_operand="SMA")
    return StrategyGene(
        id="test_gene_001",
        indicators=[indicator],
        long_entry_conditions=[condition],
        short_entry_conditions=[],
    )


def _sample_backtest_service() -> Mock:
    service = Mock()
    service.run_backtest.return_value = {
        "performance_metrics": {
            "total_return": 0.25,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "total_trades": 100,
        }
    }
    service.data_service = Mock()
    service.data_service.get_ohlcv_data.return_value = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [98, 99, 100],
            "Close": [103, 104, 105],
            "Volume": [1000, 1100, 1200],
        }
    )
    return service


def test_hybrid_evaluator_returns_fitness_without_prediction_bonus():
    from app.services.auto_strategy.core.hybrid.hybrid_individual_evaluator import (
        HybridIndividualEvaluator,
    )

    evaluator = HybridIndividualEvaluator(
        backtest_service=_sample_backtest_service(),
        predictor=Mock(),
    )
    evaluator.set_backtest_config(
        {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000.0,
        }
    )
    config = GAConfig(
        fitness_weights={
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        },
        fallback_start_date="2023-01-01",
        fallback_end_date="2023-12-31",
    )

    with patch.object(evaluator, "_get_evaluation_context", return_value={}):
        fitness = evaluator.evaluate(_sample_gene(), config)

    assert isinstance(fitness, tuple)
    assert len(fitness) == 1
    assert fitness[0] > 0


def test_hybrid_evaluator_multi_objective_does_not_support_prediction_score():
    config = GAConfig(
        enable_multi_objective=True,
        objectives=["sharpe_ratio", "prediction_score"],
        objective_weights=[1.0, 1.0],
    )
    from app.services.auto_strategy.config import ConfigValidator
    is_valid, errors = ConfigValidator.validate(config)

    assert is_valid is False
    assert any("prediction_score" in error for error in errors)
