from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.auto_strategy.genes import Condition
from app.services.auto_strategy.strategies.exit_decision_engine import (
    ExitDecisionEngine,
)


class TestExitDecisionEngine:
    @pytest.fixture
    def strategy(self):
        strategy = MagicMock()
        strategy.position = SimpleNamespace(size=1.0)
        strategy._current_bar_index = 0
        strategy._precomputed_exit_signals = {}
        strategy.condition_evaluator = MagicMock()
        strategy.gene = MagicMock()
        strategy.gene.long_exit_conditions = []
        strategy.gene.short_exit_conditions = []
        return strategy

    @pytest.fixture
    def engine(self, strategy):
        return ExitDecisionEngine(strategy)

    def test_evaluate_exit_conditions_falls_back_when_cached_signal_is_scalar(
        self, engine, strategy
    ):
        strategy._precomputed_exit_signals = {1.0: np.bool_(True)}
        strategy.gene.long_exit_conditions = [
            Condition(left_operand="close", operator=">", right_operand=100.0)
        ]
        strategy.condition_evaluator.evaluate_single_condition.return_value = True

        result = engine._evaluate_exit_conditions(strategy.gene.long_exit_conditions)

        assert result is True
        strategy.condition_evaluator.evaluate_single_condition.assert_called_once_with(
            strategy.gene.long_exit_conditions[0],
            strategy,
        )
