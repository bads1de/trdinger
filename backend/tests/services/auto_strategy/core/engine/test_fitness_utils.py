from types import SimpleNamespace

from app.services.auto_strategy.core.engine.fitness_utils import (
    extract_individual_primary_fitness,
    extract_primary_fitness_from_result,
    extract_result_fitness,
)


def test_extract_individual_primary_fitness_prefers_weighted_values():
    individual = SimpleNamespace(
        fitness=SimpleNamespace(wvalues=(-1.5,), values=(1.5,))
    )

    assert extract_individual_primary_fitness(individual) == -1.5


def test_extract_primary_fitness_from_result_handles_scalar_and_tuple():
    assert extract_primary_fitness_from_result((2.5, 0.1)) == 2.5
    assert extract_primary_fitness_from_result(1.25) == 1.25
    assert extract_primary_fitness_from_result(None) == 0.0


def test_extract_result_fitness_returns_scalar_for_single_objective():
    best = SimpleNamespace(fitness=SimpleNamespace(values=(3.0,)))

    assert extract_result_fitness(best) == 3.0


def test_extract_result_fitness_returns_tuple_for_multi_objective():
    best = SimpleNamespace(fitness=SimpleNamespace(values=(3.0, -0.4)))

    assert extract_result_fitness(best) == (3.0, -0.4)
