from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app.services.auto_strategy.genes import IndicatorGene, StrategyGene
from app.services.auto_strategy.genes.conditions import Condition
from app.services.auto_strategy.strategies.strategy_initializer import (
    StrategyInitializer,
)


class TestStrategyInitializer:
    def _build_strategy(self) -> SimpleNamespace:
        data = SimpleNamespace(
            High=np.array([105.0, 106.0, 107.0]),
            Low=np.array([95.0, 96.0, 97.0]),
            Close=np.array([100.0, 101.0, 102.0]),
            df=pd.DataFrame(
                {
                    "High": [105.0, 106.0, 107.0],
                    "Low": [95.0, 96.0, 97.0],
                    "Close": [100.0, 101.0, 102.0],
                }
            ),
        )
        data.__len__ = MagicMock(return_value=3)
        return SimpleNamespace(
            data=data,
            gene=StrategyGene(indicators=[]),
            indicator_calculator=SimpleNamespace(init_indicator=MagicMock()),
            ml_filter=SimpleNamespace(precompute_ml_features=MagicMock()),
            condition_evaluator=SimpleNamespace(
                calculate_conditions_vectorized=MagicMock(return_value=np.array([True]))
            ),
            volatility_gate_enabled=False,
            ml_predictor=None,
            _precomputed_signals={},
            _precomputed_atr="sentinel",
            _precomputed_tpsl_atr={"old": np.array([1.0])},
            _get_effective_tpsl_gene=MagicMock(return_value=None),
        )

    def test_initialize_calls_init_indicator_only_for_enabled_indicators(self):
        strategy = self._build_strategy()
        strategy.gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="EMA", parameters={"period": 10}, enabled=False),
            ]
        )

        initializer = StrategyInitializer(strategy)
        initializer.initialize()

        strategy.indicator_calculator.init_indicator.assert_called_once()
        called_gene = strategy.indicator_calculator.init_indicator.call_args.args[0]
        assert called_gene.type == "SMA"

    def test_initialize_precomputes_ml_and_condition_signals(self):
        strategy = self._build_strategy()
        strategy.volatility_gate_enabled = True
        strategy.ml_predictor = object()
        strategy.gene = StrategyGene(
            indicators=[],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand=100.0)
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand=90.0)
            ],
        )

        initializer = StrategyInitializer(strategy)
        initializer.initialize()

        strategy.ml_filter.precompute_ml_features.assert_called_once()
        assert 1.0 in strategy._precomputed_signals
        assert -1.0 in strategy._precomputed_signals
        assert strategy.condition_evaluator.calculate_conditions_vectorized.call_count == 2
