"""
GAエンジン統計計算の非有限値（±infペナルティ）耐性テスト

制約違反・低取引回数の個体には fitness として ±inf（ペナルティ）が設定される。
この値がそのまま `np.mean` / `np.std` に渡ると RuntimeWarning
（invalid value encountered in subtract）を引き起こし、警告をエラーとして
扱う環境（CI の `-W error` 等）では GA 実行全体が失敗する。
統計計算が非有限値を除外することを検証する。
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.auto_strategy.core.engine.ga_engine import GeneticAlgorithmEngine


class _MockIndividual:
    """fitness.values のみを持つ最小限の個体"""

    def __init__(self, values: tuple[float, ...]) -> None:
        self.fitness = SimpleNamespace(values=values)


class TestSafeFitnessStat:
    """``_safe_fitness_stat`` の非有限値除外ロジック"""

    def test_excludes_inf_from_stats(self) -> None:
        values = [0.5, -1.0, -math.inf, math.inf]
        assert GeneticAlgorithmEngine._safe_fitness_stat(values, np.mean) == -0.25
        assert GeneticAlgorithmEngine._safe_fitness_stat(values, np.min) == -1.0
        assert GeneticAlgorithmEngine._safe_fitness_stat(values, np.max) == 0.5
        assert math.isfinite(GeneticAlgorithmEngine._safe_fitness_stat(values, np.std))

    def test_all_infinite_returns_nan_without_error(self) -> None:
        values = [-math.inf, -math.inf]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert math.isnan(
                GeneticAlgorithmEngine._safe_fitness_stat(values, np.mean)
            )

    def test_no_warning_when_inf_present(self) -> None:
        values = [1.0, -math.inf, 2.0]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            GeneticAlgorithmEngine._safe_fitness_stat(values, np.mean)
            GeneticAlgorithmEngine._safe_fitness_stat(values, np.std)
            GeneticAlgorithmEngine._safe_fitness_stat(values, np.min)
            GeneticAlgorithmEngine._safe_fitness_stat(values, np.max)


class TestCreateStatistics:
    """DEAP Statistics が ±inf fitness を含む集団でも警告を出さないこと"""

    def _make_engine(self) -> GeneticAlgorithmEngine:
        return GeneticAlgorithmEngine.__new__(GeneticAlgorithmEngine)

    def test_compile_with_inf_fitness_no_runtime_warning(self) -> None:
        engine = self._make_engine()
        stats = engine._create_statistics()
        population = [
            _MockIndividual((0.5,)),
            _MockIndividual((-1.0,)),
            _MockIndividual((-math.inf,)),
            _MockIndividual((math.inf,)),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            record = stats.compile(population)

        assert record["min"] == -1.0
        assert record["max"] == 0.5
        assert record["avg"] == pytest.approx(-0.25)
        assert math.isfinite(record["std"])

    def test_compile_all_infinite_population(self) -> None:
        engine = self._make_engine()
        stats = engine._create_statistics()
        population = [_MockIndividual((-math.inf,)), _MockIndividual((-math.inf,))]

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            record = stats.compile(population)

        assert math.isnan(record["min"])
        assert math.isnan(record["max"])
        assert math.isnan(record["avg"])


class TestProcessResultsSanitization:
    """±inf fitness が結果出力（best_fitness / fitness_scores / pareto）へ漏れないこと"""

    def test_inf_fitness_sanitized_in_result_output(self) -> None:
        engine = GeneticAlgorithmEngine.__new__(GeneticAlgorithmEngine)
        engine.result_processor = MagicMock()
        engine.parameter_tuning_manager = MagicMock()
        engine.parameter_tuning_manager.build_individual_evaluation_summary.return_value = None

        class MockInd:
            def __init__(self, fitness_values: tuple[float, ...], ind_id: str) -> None:
                self.fitness = SimpleNamespace(values=fitness_values)
                self.id = ind_id

        population = [
            MockInd((-math.inf,), "penalized"),
            MockInd((0.5,), "normal"),
        ]
        best_individual = population[0]
        best_gene = object()
        best_strategies = [{"strategy": object(), "fitness_values": [-math.inf]}]

        engine.result_processor.extract_best_individuals.return_value = (
            best_individual,
            best_gene,
            best_strategies,
        )
        engine.result_processor.sort_population.return_value = population
        engine.result_processor.get_strategy_result_key.side_effect = lambda ind: ind.id

        config = SimpleNamespace(generations=2, objectives=["weighted_score"])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = engine._process_results(
                population, config, logbook=None, start_time=0.0
            )

        # -inf は None / 有限値に置換され、JSON 安全になっている
        assert result["best_fitness"] is None
        assert result["fitness_scores"] == [0.0, 0.5]
        assert result["pareto_front"][0]["fitness_values"] == [0.0]

        import json

        json.dumps(
            {
                "best_fitness": result["best_fitness"],
                "fitness_scores": result["fitness_scores"],
                "pareto_front": [
                    {"fitness_values": p["fitness_values"]}
                    for p in result["pareto_front"]
                ],
            },
            allow_nan=False,
        )
