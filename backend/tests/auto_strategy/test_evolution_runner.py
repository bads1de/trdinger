"""Tests for EvolutionRunner dynamic objective weighting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence
from unittest.mock import Mock

import numpy as np

from app.services.auto_strategy.core.evolution_runner import EvolutionRunner


class _DummyFitness:
    def __init__(self, values: Sequence[float]):
        self.values = tuple(values)
        self.valid = True


class _DummyIndividual(list):
    def __init__(self, values: Sequence[float]):
        super().__init__(values)
        self.fitness = _DummyFitness(values)


def test_dynamic_objective_scalars_emphasize_risk_metrics() -> None:
    """Risk metrics receive boosted scaling factors when averages are high."""

    runner = EvolutionRunner(toolbox=Mock(), stats=None)

    config = SimpleNamespace(
        dynamic_objective_reweighting=True,
        objectives=[
            "total_return",
            "max_drawdown",
            "ulcer_index",
            "trade_frequency_penalty",
        ],
        objective_dynamic_scalars={},
    )

    population = [
        _DummyIndividual((0.5, 0.2, 0.15, 0.4)),
        _DummyIndividual((0.4, 0.1, 0.05, 0.2)),
    ]

    runner._update_dynamic_objective_scalars(population, config)

    assert np.isclose(config.objective_dynamic_scalars["max_drawdown"], 1.15)
    assert np.isclose(config.objective_dynamic_scalars["ulcer_index"], 1.1)
    assert np.isclose(config.objective_dynamic_scalars["trade_frequency_penalty"], 1.3)
    assert config.objective_dynamic_scalars.get("total_return", 1.0) == 1.0
