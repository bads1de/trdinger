"""
ParameterTuningManagerのユニットテスト

``app.services.auto_strategy.core.engine.parameter_tuning_manager.ParameterTuningManager`` の
ヘルパーメソッド（候補選択、フィットネス抽出、full fidelity評価）を検証します。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.services.auto_strategy.config.ga.ga_config import GAConfig
from app.services.auto_strategy.core.engine.parameter_tuning_manager import (
    ParameterTuningManager,
)
from app.services.auto_strategy.genes import StrategyGene


def _make_strategy_individual(fitness_values: tuple[float, ...], gene_id: str = "x") -> Mock:
    """StrategyGene として振る舞う Mock 個体"""
    ind = Mock()
    ind.__class__ = StrategyGene
    ind.fitness.values = fitness_values
    ind.fitness.valid = True
    ind.id = gene_id
    return ind


def _make_gene(gene_id: str = "x") -> StrategyGene:
    """実際の StrategyGene を返す"""
    gene = StrategyGene.create_default()
    gene.id = gene_id
    return gene


class TestParameterTuningManagerInit:
    """``__init__`` のテスト"""

    def test_stores_individual_evaluator(self) -> None:
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        assert manager.individual_evaluator is evaluator


class TestSelectTuningCandidates:
    """``select_tuning_candidates`` の挙動テスト"""

    def test_empty_population_returns_empty(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        config = GAConfig()
        result = manager.select_tuning_candidates(
            population=[], config=config, fallback_gene=None
        )
        assert result == []

    def test_uses_default_budget_from_tuning_config(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        config = GAConfig()
        # Default tuning_config.elite_count = 3
        individuals = [
            _make_strategy_individual((float(i),), gene_id=f"g{i}")
            for i in range(10)
        ]
        result = manager.select_tuning_candidates(
            population=individuals, config=config
        )
        # Should return at most elite_count (3)
        assert len(result) <= 3

    def test_uses_custom_budget_from_tuning_config(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        config = GAConfig()
        # Override the tuning_config
        config.tuning_config.elite_count = 5
        individuals = [
            _make_strategy_individual((float(i),), gene_id=f"g{i}")
            for i in range(20)
        ]
        result = manager.select_tuning_candidates(
            population=individuals, config=config
        )
        assert len(result) == 5

    def test_clamps_budget_to_minimum_one(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        config = GAConfig()
        config.tuning_config.elite_count = 0
        individuals = [
            _make_strategy_individual((0.1,), gene_id="a"),
            _make_strategy_individual((0.5,), gene_id="b"),
        ]
        result = manager.select_tuning_candidates(
            population=individuals, config=config
        )
        # Budget clamped to max(1, 0) = 1
        assert len(result) == 1

    def test_handles_invalid_budget_falls_back_to_one(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        config = Mock()
        config.tuning_config = SimpleNamespace(elite_count="not a number")

        individuals = [
            _make_strategy_individual((0.1,), gene_id="a"),
            _make_strategy_individual((0.5,), gene_id="b"),
        ]
        result = manager.select_tuning_candidates(
            population=individuals, config=config
        )
        # budget falls back to 1
        assert len(result) == 1

    def test_skips_non_strategy_gene_individuals(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        config = GAConfig()
        config.tuning_config.elite_count = 3

        good = _make_strategy_individual((0.5,), gene_id="g1")
        bad = MagicMock()  # not StrategyGene

        result = manager.select_tuning_candidates(
            population=[good, bad], config=config
        )
        # Only the StrategyGene individual should be included
        assert len(result) == 1
        assert result[0] is good

    def test_deduplicates_by_identity(self) -> None:
        """同じ id の個体は1つだけ選択される"""
        manager = ParameterTuningManager(individual_evaluator=Mock())
        config = GAConfig()
        config.tuning_config.elite_count = 5

        ind1 = _make_strategy_individual((0.1,), gene_id="dup")
        ind2 = _make_strategy_individual((0.2,), gene_id="dup")
        ind3 = _make_strategy_individual((0.3,), gene_id="other")

        result = manager.select_tuning_candidates(
            population=[ind1, ind2, ind3], config=config
        )
        # Only 2 unique ids
        assert len(result) == 2

    def test_uses_fallback_when_no_candidates(self) -> None:
        """候補が見つからない場合、fallback_gene が使われる"""
        manager = ParameterTuningManager(individual_evaluator=Mock())
        config = GAConfig()

        fallback = _make_gene("fallback")

        result = manager.select_tuning_candidates(
            population=[], config=config, fallback_gene=fallback
        )
        assert result == [fallback]

    def test_does_not_use_fallback_when_candidates_exist(self) -> None:
        """候補が既に存在する場合、fallback_gene は使われない"""
        manager = ParameterTuningManager(individual_evaluator=Mock())
        config = GAConfig()
        ind = _make_strategy_individual((0.5,), gene_id="g1")
        fallback = _make_gene("fallback")

        result = manager.select_tuning_candidates(
            population=[ind], config=config, fallback_gene=fallback
        )
        assert fallback not in result

    def test_handles_missing_tuning_config(self) -> None:
        """config に tuning_config が無い場合、デフォルトバジェット1を使用"""
        manager = ParameterTuningManager(individual_evaluator=Mock())
        config = Mock()
        # No tuning_config attribute
        del config.tuning_config  # ensure AttributeError
        ind = _make_strategy_individual((0.5,), gene_id="g1")
        result = manager.select_tuning_candidates(population=[ind], config=config)
        assert len(result) >= 1


class TestEvaluateIndividualWithFullFidelity:
    """``evaluate_individual_with_full_fidelity`` の挙動テスト"""

    def test_multi_fidelity_calls_with_force_refresh(self) -> None:
        """multi-fidelity 有効時、force_refresh=True で evaluator.evaluate を呼ぶ"""
        evaluator = Mock()
        evaluator.evaluate.return_value = (1.0, 0.5)
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        ind = _make_strategy_individual((0.5,))

        config = Mock()
        # is_multi_fidelity_enabled returns True for this config
        with patch(
            "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_multi_fidelity_enabled",
            return_value=True,
        ):
            result = manager.evaluate_individual_with_full_fidelity(ind, config)

        assert result == (1.0, 0.5)
        evaluator.evaluate.assert_called_once_with(ind, config, force_refresh=True)

    def test_single_fidelity_calls_without_force_refresh(self) -> None:
        """multi-fidelity 無効時、force_refresh なしで evaluator.evaluate を呼ぶ"""
        evaluator = Mock()
        evaluator.evaluate.return_value = (0.8,)
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        ind = _make_strategy_individual((0.5,))

        config = Mock()
        with patch(
            "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_multi_fidelity_enabled",
            return_value=False,
        ):
            result = manager.evaluate_individual_with_full_fidelity(ind, config)

        assert result == (0.8,)
        evaluator.evaluate.assert_called_once_with(ind, config)


class TestExtractPrimaryFitnessFromResult:
    """``extract_primary_fitness_from_result`` (staticmethod) のテスト"""

    def test_scalar_result(self) -> None:
        result = ParameterTuningManager.extract_primary_fitness_from_result(1.5)
        assert result == 1.5

    def test_tuple_result(self) -> None:
        result = ParameterTuningManager.extract_primary_fitness_from_result((2.0, 0.5))
        assert result == 2.0

    def test_list_result(self) -> None:
        result = ParameterTuningManager.extract_primary_fitness_from_result([0.7, 0.3])
        assert result == 0.7

    def test_none_result(self) -> None:
        result = ParameterTuningManager.extract_primary_fitness_from_result(None)
        assert result == 0.0

    def test_negative_fitness(self) -> None:
        result = ParameterTuningManager.extract_primary_fitness_from_result(-0.5)
        assert result == -0.5


class TestSelectBestTunedCandidateByFitness:
    """``select_best_tuned_candidate_by_fitness`` のテスト"""

    def test_empty_candidates_returns_none(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        config = GAConfig()
        result = manager.select_best_tuned_candidate_by_fitness(
            tuned_candidates=[], config=config
        )
        assert result is None

    def test_picks_highest_primary_fitness(self) -> None:
        """最高 primary fitness の候補を選ぶ"""
        evaluator = Mock()

        def mock_evaluate(ind, config, **kwargs):
            # First individual returns 0.5, second returns 0.9
            if ind is candidates[0]:
                return (0.5,)
            return (0.9,)

        evaluator.evaluate.side_effect = mock_evaluate
        manager = ParameterTuningManager(individual_evaluator=evaluator)

        c1 = _make_gene("c1")
        c2 = _make_gene("c2")
        candidates = [c1, c2]

        with patch.object(
            manager, "build_individual_evaluation_summary", return_value={"mode": "single"}
        ):
            result = manager.select_best_tuned_candidate_by_fitness(
                tuned_candidates=candidates, config=GAConfig()
            )

        assert result is not None
        best_gene, best_fitness, summary = result
        assert best_gene is c2
        assert best_fitness == 0.9

    def test_continues_when_evaluation_fails(self) -> None:
        """1つの候補で評価が失敗しても、他の候補は評価される"""
        evaluator = Mock()
        call_count = [0]

        def mock_evaluate(ind, config, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Eval failed")
            return (0.7,)

        evaluator.evaluate.side_effect = mock_evaluate
        manager = ParameterTuningManager(individual_evaluator=evaluator)

        c1 = _make_gene("c1")
        c2 = _make_gene("c2")

        with patch.object(
            manager, "build_individual_evaluation_summary", return_value={"mode": "single"}
        ):
            result = manager.select_best_tuned_candidate_by_fitness(
                tuned_candidates=[c1, c2], config=GAConfig()
            )

        # Should return c2 (only one that succeeded)
        assert result is not None
        assert result[0] is c2

    def test_returns_none_if_all_evaluations_fail(self) -> None:
        """全ての評価が失敗した場合 None"""
        evaluator = Mock()
        evaluator.evaluate.side_effect = RuntimeError("All failed")
        manager = ParameterTuningManager(individual_evaluator=evaluator)

        candidates = [_make_gene("c1"), _make_gene("c2")]
        result = manager.select_best_tuned_candidate_by_fitness(
            tuned_candidates=candidates, config=GAConfig()
        )
        assert result is None


class TestTuneEliteParameters:
    """``tune_elite_parameters`` のテスト"""

    def test_returns_tuned_gene_on_success(self) -> None:
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        best_gene = _make_gene("best")
        tuned_gene = _make_gene("tuned")

        with patch(
            "app.services.auto_strategy.optimization.StrategyParameterTuner"
        ) as MockTuner:
            MockTuner.from_ga_config.return_value.tune.return_value = tuned_gene

            result = manager.tune_elite_parameters(best_gene, GAConfig())

        assert result is tuned_gene

    def test_returns_original_gene_on_exception(self) -> None:
        """チューニング中に例外が発生した場合、元の遺伝子を返す"""
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        best_gene = _make_gene("best")

        with patch(
            "app.services.auto_strategy.optimization.StrategyParameterTuner"
        ) as MockTuner:
            MockTuner.from_ga_config.side_effect = RuntimeError("Tuner init failed")

            result = manager.tune_elite_parameters(best_gene, GAConfig())

        # Falls back to original gene
        assert result is best_gene
