from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app.services.auto_strategy.config import objective_registry
from app.services.auto_strategy.core.engine.evolution_runner import EvolutionRunner


class _DummyIndividual:
    def __init__(self, values):
        self.fitness = SimpleNamespace(valid=True, values=tuple(values))


def test_dynamic_objective_scalars_use_objective_registry(monkeypatch):
    monkeypatch.setattr(
        objective_registry,
        "is_dynamic_scalar_objective",
        lambda objective: objective == "custom_loss",
    )

    runner = EvolutionRunner(toolbox=Mock(), stats=None)
    config = SimpleNamespace(
        dynamic_objective_reweighting=True,
        objectives=["custom_loss", "total_return"],
        objective_dynamic_scalars={},
    )
    population = [
        _DummyIndividual((0.5, 0.1)),
        _DummyIndividual((0.5, 0.3)),
    ]

    runner._update_dynamic_objective_scalars(population, config)

    assert np.isclose(config.objective_dynamic_scalars["custom_loss"], 1.5)
    assert config.objective_dynamic_scalars["total_return"] == 1.0
