"""Tests for EvolutionRunner."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from app.services.auto_strategy.core.evolution_runner import EvolutionRunner


class _DummyFitness:
    def __init__(self, values: Sequence[float] | None = None):
        self._valid = False
        if values is not None:
            self.values = tuple(values)

    @property
    def values(self):
        if not hasattr(self, "_values"):
            raise AttributeError("Fitness has no attribute 'values'")
        return self._values

    @values.setter
    def values(self, v):
        self._values = tuple(v)
        self._valid = True

    @values.deleter
    def values(self):
        if hasattr(self, "_values"):
            del self._values
        self._valid = False

    @property
    def valid(self):
        return self._valid

    @valid.setter
    def valid(self, v):
        self._valid = v


class _DummyIndividual(list):
    def __init__(self, values: Sequence[float] | None = None):
        super().__init__([])
        self.fitness = _DummyFitness(values)


class TestEvolutionRunner:
    @pytest.fixture
    def mock_toolbox(self):
        toolbox = Mock()
        # DEAP toolbox operations
        toolbox.map = Mock(side_effect=map)  # Default map
        toolbox.clone = Mock(
            side_effect=lambda x: _DummyIndividual(
                x.fitness.values if x.fitness.valid else None
            )
        )
        toolbox.mate = Mock()
        toolbox.mutate = Mock()
        toolbox.select = Mock(side_effect=lambda pop, k: pop[:k])
        toolbox.evaluate = Mock(return_value=(1.0,))
        return toolbox

    @pytest.fixture
    def mock_stats(self):
        stats = Mock()
        stats.compile = Mock(return_value={"avg": 1.0})
        return stats

    @pytest.fixture
    def dummy_population(self):
        # Create a population of 4 individuals
        return [
            _DummyIndividual((1.0,)),
            _DummyIndividual((0.8,)),
            _DummyIndividual((0.5,)),
            _DummyIndividual((0.2,)),
        ]

    @pytest.fixture
    def config(self):
        return SimpleNamespace(
            generations=2,
            objectives=["total_return"],
            crossover_rate=0.5,
            mutation_rate=0.2,
            enable_fitness_sharing=False,
            dynamic_objective_reweighting=False,
        )

    def test_run_evolution_basic_flow(
        self, mock_toolbox, mock_stats, dummy_population, config
    ):
        """Test the basic evolution flow (sequential evaluation)."""
        runner = EvolutionRunner(toolbox=mock_toolbox, stats=mock_stats)

        # Mock random to trigger crossover/mutation deterministically
        with patch("random.random", return_value=0.1):  # < 0.5 (cx) and < 0.2 (mut)
            result_pop, logbook = runner.run_evolution(dummy_population, config)

        assert len(result_pop) == len(dummy_population)
        assert mock_toolbox.evaluate.call_count > 0
        assert mock_toolbox.mate.called
        assert mock_toolbox.mutate.called
        assert mock_toolbox.select.called
        assert mock_stats.compile.call_count == config.generations
        assert len(logbook) == config.generations

    def test_evaluate_population_sequential(self, mock_toolbox, dummy_population):
        """Test sequential population evaluation."""
        runner = EvolutionRunner(toolbox=mock_toolbox, stats=None)

        # Reset fitnesses
        for ind in dummy_population:
            ind.fitness.valid = False
            ind.fitness.values = ()

        runner._evaluate_population(dummy_population)

        # Toolbox evaluate should be called for each individual
        assert mock_toolbox.evaluate.call_count == len(dummy_population)
        for ind in dummy_population:
            assert ind.fitness.valid
            assert ind.fitness.values == (1.0,)

    def test_evaluate_population_parallel(self, mock_toolbox, dummy_population):
        """Test parallel population evaluation."""
        mock_parallel_evaluator = Mock()
        # Return list of fitness tuples
        mock_parallel_evaluator.evaluate_population.return_value = [
            (2.0,) for _ in dummy_population
        ]

        runner = EvolutionRunner(
            toolbox=mock_toolbox, stats=None, parallel_evaluator=mock_parallel_evaluator
        )

        # Reset fitnesses
        for ind in dummy_population:
            ind.fitness.valid = False

        runner._evaluate_population(dummy_population)

        # Parallel evaluator should be called once with full population
        mock_parallel_evaluator.evaluate_population.assert_called_once_with(
            dummy_population
        )

        # Toolbox evaluate should NOT be called
        mock_toolbox.evaluate.assert_not_called()

        # Fitnesses should be updated
        for ind in dummy_population:
            assert ind.fitness.valid
            assert ind.fitness.values == (2.0,)

    def test_evaluate_invalid_individuals_mixed(self, mock_toolbox, dummy_population):
        """Test evaluation of only invalid individuals."""
        runner = EvolutionRunner(toolbox=mock_toolbox, stats=None)

        # Make half valid, half invalid
        dummy_population[0].fitness.valid = True
        dummy_population[0].fitness.values = (5.0,)
        dummy_population[1].fitness.valid = False

        invalid_ind = [ind for ind in dummy_population if not ind.fitness.valid]

        runner._evaluate_invalid_individuals(invalid_ind)

        # Only invalid individuals should be evaluated
        assert mock_toolbox.evaluate.call_count == len(invalid_ind)

        # Check values
        assert dummy_population[0].fitness.values == (5.0,)  # Should not change
        assert dummy_population[1].fitness.values == (1.0,)  # Should be updated

    def test_fitness_sharing_application(self, mock_toolbox, dummy_population, config):
        """Test fitness sharing application during evolution."""
        config.enable_fitness_sharing = True
        mock_fitness_sharing = Mock()
        mock_fitness_sharing.apply_fitness_sharing.side_effect = (
            lambda pop: pop
        )  # Identity

        runner = EvolutionRunner(
            toolbox=mock_toolbox, stats=None, fitness_sharing=mock_fitness_sharing
        )

        with patch("random.random", return_value=0.9):  # No cx/mut
            runner.run_evolution(dummy_population, config)

        # Fitness sharing should be applied in each generation
        assert (
            mock_fitness_sharing.apply_fitness_sharing.call_count == config.generations
        )

    def test_dynamic_objective_scalars_emphasize_risk_metrics(self) -> None:
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
        assert np.isclose(
            config.objective_dynamic_scalars["trade_frequency_penalty"], 1.3
        )
        assert config.objective_dynamic_scalars.get("total_return", 1.0) == 1.0

    def test_hall_of_fame_update(self, mock_toolbox, dummy_population, config):
        """Test that Hall of Fame is updated."""
        mock_hof = Mock()

        runner = EvolutionRunner(toolbox=mock_toolbox, stats=None)

        with patch("random.random", return_value=0.9):
            runner.run_evolution(dummy_population, config, halloffame=mock_hof)

        # HOF update called at start + each generation
        assert mock_hof.update.call_count == config.generations + 1




