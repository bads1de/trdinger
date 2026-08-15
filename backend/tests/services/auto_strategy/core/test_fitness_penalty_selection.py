"""
有限ペナルティ fitness と DEAP NSGA-II 選択挙動の検証テスト

``get_penalty_values`` が返す ±inf を有限の大きな値
（``PENALTY_FITNESS_MAGNITUDE``）へ変更しても、実現可能な fitness の範囲を
大きく外れた値であるため、DEAP の NSGA-II 選択では任意の有効個体に支配され、
±inf と同等の順序関係・選択結果になることを検証する。
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

from deap import base, creator, tools

from app.services.auto_strategy.config.constants import PENALTY_FITNESS_MAGNITUDE
from app.services.auto_strategy.core.fitness.fitness_calculator import FitnessCalculator


def _make_fitness_class(weights: tuple[float, ...]):
    """一意な名前で DEAP Fitness クラスを作成する（creator はグローバルのため）。"""
    name = f"FitnessPenalty_{uuid.uuid4().hex[:8]}"
    creator.create(name, base.Fitness, weights=weights)
    return getattr(creator, name)


def _make_individual(fitness_class, values: tuple[float, ...]):
    individual = SimpleNamespace(fitness=fitness_class())
    individual.fitness.values = values
    return individual


class TestPenaltyValuesFinite:
    """ペナルティ値が有限で JSON 安全であること"""

    def test_penalty_values_use_finite_magnitude(self) -> None:
        calculator = FitnessCalculator()
        config = SimpleNamespace(objectives=["total_return", "max_drawdown"])

        penalty = calculator.get_penalty_values(config)

        assert penalty == (
            -PENALTY_FITNESS_MAGNITUDE,
            PENALTY_FITNESS_MAGNITUDE,
        )
        assert all(
            value != float("inf") and value != -float("inf") for value in penalty
        )

    def test_penalty_values_are_json_safe(self) -> None:
        calculator = FitnessCalculator()
        config = SimpleNamespace(objectives=["total_return", "max_drawdown"])

        penalty = calculator.get_penalty_values(config)

        # Starlette の JSONResponse 相当（allow_nan=False）で例外にならない
        json.dumps(list(penalty), allow_nan=False)


class TestDeapSelectionEquivalence:
    """有限ペナルティが ±inf と同等の選択挙動になること"""

    def test_feasible_individual_dominates_penalized(self) -> None:
        fitness_class = _make_fitness_class((1.0,))
        feasible = _make_individual(fitness_class, (0.5,))
        penalized_finite = _make_individual(
            fitness_class, (-PENALTY_FITNESS_MAGNITUDE,)
        )
        penalized_inf = _make_individual(fitness_class, (-float("inf"),))

        assert feasible.fitness.dominates(penalized_finite.fitness)
        assert feasible.fitness.dominates(penalized_inf.fitness)
        assert not penalized_finite.fitness.dominates(feasible.fitness)
        assert not penalized_inf.fitness.dominates(feasible.fitness)

    def test_nsga2_selection_excludes_penalized(self) -> None:
        """ペナルティ個体は有効個体がいる限り NSGA-II の選択結果に含まれない"""
        fitness_class = _make_fitness_class((1.0,))
        for penalty in (-PENALTY_FITNESS_MAGNITUDE, -float("inf")):
            population = [
                _make_individual(fitness_class, (penalty,)),
                _make_individual(fitness_class, (0.8,)),
                _make_individual(fitness_class, (0.3,)),
                _make_individual(fitness_class, (penalty,)),
            ]

            selected = tools.selNSGA2(population, 2)

            assert len(selected) == 2
            assert all(
                ind.fitness.values[0] > -PENALTY_FITNESS_MAGNITUDE / 2
                for ind in selected
            ), "ペナルティ個体が選択されてはいけない"

    def test_nsga2_top_selection_same_for_finite_and_inf_penalty(self) -> None:
        """有限ペナルティと ±inf ペナルティで選択結果（上位k件）が一致する"""
        fitness_class = _make_fitness_class((1.0,))

        results: dict[float, list[float]] = {}
        for penalty in (-PENALTY_FITNESS_MAGNITUDE, -float("inf")):
            population = [
                _make_individual(fitness_class, (0.8,)),
                _make_individual(fitness_class, (0.3,)),
                _make_individual(fitness_class, (-0.2,)),
                _make_individual(fitness_class, (1.2,)),
                _make_individual(fitness_class, (penalty,)),
                _make_individual(fitness_class, (penalty,)),
            ]

            selected = tools.selNSGA2(population, 4)
            results[penalty] = sorted(ind.fitness.values[0] for ind in selected)

        assert results[-PENALTY_FITNESS_MAGNITUDE] == results[-float("inf")]
        assert results[-PENALTY_FITNESS_MAGNITUDE] == [-0.2, 0.3, 0.8, 1.2]

    def test_multi_objective_penalized_is_dominated(self) -> None:
        """最大化 + 最小化の多目的でも有限ペナルティ個体は有効個体に支配される"""
        fitness_class = _make_fitness_class(
            (1.0, -1.0)
        )  # total_return 最大化, max_drawdown 最小化
        feasible = _make_individual(fitness_class, (0.1, 0.2))
        penalized_finite = _make_individual(
            fitness_class, (-PENALTY_FITNESS_MAGNITUDE, PENALTY_FITNESS_MAGNITUDE)
        )
        penalized_inf = _make_individual(fitness_class, (-float("inf"), float("inf")))

        assert feasible.fitness.dominates(penalized_finite.fitness)
        assert feasible.fitness.dominates(penalized_inf.fitness)

    def test_nsga2_all_penalized_population_does_not_crash(self) -> None:
        """全個体がペナルティでも NSGA-II が正常に選択できる"""
        fitness_class = _make_fitness_class((1.0,))
        population = [
            _make_individual(fitness_class, (-PENALTY_FITNESS_MAGNITUDE,)),
            _make_individual(fitness_class, (-PENALTY_FITNESS_MAGNITUDE,)),
        ]

        selected = tools.selNSGA2(population, 2)

        assert len(selected) == 2
