import math
from types import SimpleNamespace

import pytest

from app.services.auto_strategy.core.engine.fitness_utils import (
    extract_individual_primary_fitness,
    extract_primary_fitness_from_result,
    extract_result_fitness,
    sanitize_fitness_for_output,
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


class TestSanitizeFitnessForOutput:
    """出力境界での ±inf / NaN の安全化テスト"""

    def test_keeps_finite_scalar(self):
        assert sanitize_fitness_for_output(1.5) == 1.5
        assert sanitize_fitness_for_output(0.0) == 0.0

    def test_replaces_inf_scalar_with_none(self):
        assert sanitize_fitness_for_output(-math.inf) is None
        assert sanitize_fitness_for_output(math.inf) is None

    def test_replaces_nan_scalar_with_none(self):
        assert sanitize_fitness_for_output(math.nan) is None

    def test_replaces_non_finite_tuple_elements_with_default(self):
        assert sanitize_fitness_for_output((1.0, -math.inf)) == (1.0, 0.0)
        assert sanitize_fitness_for_output((math.inf,)) == 0.0

    def test_empty_sequence_returns_none(self):
        assert sanitize_fitness_for_output([]) is None

    @pytest.mark.parametrize(
        "value",
        [-math.inf, math.inf, math.nan, (math.inf,), (-math.inf, 0.5)],
    )
    def test_output_is_json_safe(self, value):
        """Starlette の allow_nan=False 相当の json.dumps で例外にならない"""
        import json

        sanitized = sanitize_fitness_for_output(value)
        json.dumps(sanitized, allow_nan=False)
