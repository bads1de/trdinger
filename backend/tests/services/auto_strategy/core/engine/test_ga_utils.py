"""
GAユーティリティ関数のユニットテスト

``app.services.auto_strategy.core.engine.ga_utils`` モジュールのヘルパー関数群
（交叉、突然変異、キャッシュ無効化、フィットネス設定、DEAPラッパー生成）を
検証します。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.auto_strategy.core.engine.ga_utils import (
    _invalidate_individual_cache,
    _set_fitness_values,
    create_deap_mutate_wrapper,
    crossover_strategy_genes,
    mutate_strategy_gene,
)
from app.services.auto_strategy.genes import StrategyGene


class _StrategyGeneSubclass(type):
    """crossover classmethod を持つテスト用メタクラス"""

    def crossover(cls, parent1, parent2, config):
        return (parent1, parent2)


class _CrossableGene(metaclass=_StrategyGeneSubclass):
    """テスト用最小限のクラス（crossover のみ定義）"""


def _make_gene_with_methods() -> MagicMock:
    """mutate/adaptive_mutate を持つモック遺伝子"""
    gene = MagicMock(spec=StrategyGene)
    gene.id = "test-gene"
    return gene


class TestCrossoverStrategyGenes:
    """``crossover_strategy_genes`` の挙動テスト"""

    def test_delegates_to_type_crossover_classmethod(self) -> None:
        # 実際のクラスを使い、crossover を patch
        parent1 = _CrossableGene()
        parent2 = _CrossableGene()
        config = SimpleNamespace()

        with patch.object(_CrossableGene, "crossover", return_value=(parent1, parent2)) as mock_co:
            result = crossover_strategy_genes(parent1, parent2, config)

        assert result == (parent1, parent2)
        mock_co.assert_called_once_with(parent1, parent2, config)

    def test_passes_config_through(self) -> None:
        parent1 = _CrossableGene()
        parent2 = _CrossableGene()
        config = SimpleNamespace(mutation_rate=0.05)

        with patch.object(_CrossableGene, "crossover", return_value=(parent1, parent2)) as mock_co:
            crossover_strategy_genes(parent1, parent2, config)
            args, _ = mock_co.call_args
            assert args[2] is config

    def test_uses_class_method_not_instance(self) -> None:
        """crossover はクラスメソッドなので、type(parent1).crossover が呼ばれる"""
        parent1 = _CrossableGene()
        parent2 = _CrossableGene()
        config = SimpleNamespace()

        with patch.object(_CrossableGene, "crossover", return_value=(parent1, parent2)) as mock_co:
            crossover_strategy_genes(parent1, parent2, config)
            # クラスメソッドとして呼ばれている
            assert mock_co.called


class TestMutateStrategyGene:
    """``mutate_strategy_gene`` の挙動テスト"""

    def test_uses_default_mutation_rate(self) -> None:
        gene = _make_gene_with_methods()
        mutated = _make_gene_with_methods()
        gene.mutate.return_value = mutated
        config = SimpleNamespace()

        result = mutate_strategy_gene(gene, config)

        assert result is mutated
        gene.mutate.assert_called_once_with(config, 0.1)

    def test_uses_custom_mutation_rate(self) -> None:
        gene = _make_gene_with_methods()
        mutated = _make_gene_with_methods()
        gene.mutate.return_value = mutated
        config = SimpleNamespace()

        mutate_strategy_gene(gene, config, mutation_rate=0.25)

        gene.mutate.assert_called_once_with(config, 0.25)

    def test_zero_mutation_rate(self) -> None:
        gene = _make_gene_with_methods()
        mutated = _make_gene_with_methods()
        gene.mutate.return_value = mutated
        config = SimpleNamespace()

        mutate_strategy_gene(gene, config, mutation_rate=0.0)

        gene.mutate.assert_called_once_with(config, 0.0)

    def test_high_mutation_rate(self) -> None:
        gene = _make_gene_with_methods()
        mutated = _make_gene_with_methods()
        gene.mutate.return_value = mutated
        config = SimpleNamespace()

        mutate_strategy_gene(gene, config, mutation_rate=1.0)

        gene.mutate.assert_called_once_with(config, 1.0)


class TestInvalidateIndividualCache:
    """``_invalidate_individual_cache`` の挙動テスト"""

    def test_deletes_fitness_values(self) -> None:
        individual = SimpleNamespace(
            fitness=SimpleNamespace(values=(1.0,)),
        )

        _invalidate_individual_cache(individual)

        assert not hasattr(individual.fitness, "values")

    def test_deletes_feature_vector(self) -> None:
        individual = SimpleNamespace(
            fitness=SimpleNamespace(values=(1.0,)),
            _feature_vector=[0.1, 0.2, 0.3],
        )

        _invalidate_individual_cache(individual)

        assert not hasattr(individual, "_feature_vector")

    def test_handles_missing_fitness_values(self) -> None:
        """fitness.values が無い場合、AttributeError を無視する"""
        individual = SimpleNamespace(fitness=SimpleNamespace())

        # Should not raise
        _invalidate_individual_cache(individual)

    def test_handles_missing_feature_vector(self) -> None:
        """_feature_vector が無い場合、AttributeError を無視する"""
        individual = SimpleNamespace(fitness=SimpleNamespace(values=(1.0,)))

        # Should not raise
        _invalidate_individual_cache(individual)

    def test_handles_missing_fitness_attribute(self) -> None:
        """fitness 自体が無い場合、AttributeError を無視する"""
        individual = SimpleNamespace()

        # Should not raise
        _invalidate_individual_cache(individual)

    def test_deletes_both_when_both_present(self) -> None:
        """両方が存在する場合、両方とも削除される"""
        individual = SimpleNamespace(
            fitness=SimpleNamespace(values=(1.0,)),
            _feature_vector=[0.1],
        )

        _invalidate_individual_cache(individual)

        assert not hasattr(individual.fitness, "values")
        assert not hasattr(individual, "_feature_vector")


class TestSetFitnessValues:
    """``_set_fitness_values`` の挙動テスト"""

    def test_sets_values_for_each_individual(self) -> None:
        individuals = [
            SimpleNamespace(fitness=SimpleNamespace()),
            SimpleNamespace(fitness=SimpleNamespace()),
        ]
        fitnesses = [(1.0,), (2.0,)]

        _set_fitness_values(individuals, fitnesses)

        assert individuals[0].fitness.values == (1.0,)
        assert individuals[1].fitness.values == (2.0,)

    def test_sets_multi_objective_fitness(self) -> None:
        individuals = [SimpleNamespace(fitness=SimpleNamespace())]
        fitnesses = [(1.0, 0.5, -0.3)]

        _set_fitness_values(individuals, fitnesses)

        assert individuals[0].fitness.values == (1.0, 0.5, -0.3)

    def test_raises_on_length_mismatch(self) -> None:
        individuals = [SimpleNamespace(fitness=SimpleNamespace())]
        fitnesses = [(1.0,), (2.0,)]

        with pytest.raises(ValueError, match="一致しません"):
            _set_fitness_values(individuals, fitnesses)

    def test_raises_on_empty_individuals_with_nonempty_fitnesses(self) -> None:
        with pytest.raises(ValueError):
            _set_fitness_values([], [(1.0,)])

    def test_raises_on_individuals_longer_than_fitnesses(self) -> None:
        individuals = [SimpleNamespace(fitness=SimpleNamespace())] * 3
        fitnesses = [(1.0,)]

        with pytest.raises(ValueError):
            _set_fitness_values(individuals, fitnesses)

    def test_empty_lists_are_ok(self) -> None:
        # Both empty -> no error
        _set_fitness_values([], [])


class TestCreateDeapMutateWrapper:
    """``create_deap_mutate_wrapper`` の挙動テスト"""

    def test_creates_callable(self) -> None:
        config = SimpleNamespace(mutation_rate=0.1)
        wrapper = create_deap_mutate_wrapper(StrategyGene, None, config)
        assert callable(wrapper)

    def test_uses_plain_mutate_when_population_is_none(self) -> None:
        config = SimpleNamespace(mutation_rate=0.15)
        gene = _make_gene_with_methods()
        mutated = _make_gene_with_methods()
        gene.mutate.return_value = mutated
        gene.adaptive_mutate = MagicMock()  # ensure it's not used

        with patch(
            "app.services.auto_strategy.core.engine.ga_utils.GeneticUtils.extract_gene_params",
            return_value={"id": "mutated-id"},
        ):
            wrapper = create_deap_mutate_wrapper(StrategyGene, None, config)
            result = wrapper(gene)

        # Returns a tuple of one individual
        assert isinstance(result, tuple)
        assert len(result) == 1
        gene.mutate.assert_called_once_with(config, mutation_rate=0.15)
        gene.adaptive_mutate.assert_not_called()

    def test_uses_adaptive_mutate_when_population_given(self) -> None:
        config = SimpleNamespace(mutation_rate=0.1)
        gene = _make_gene_with_methods()
        mutated = _make_gene_with_methods()
        population = [_make_gene_with_methods(), _make_gene_with_methods()]
        gene.adaptive_mutate.return_value = mutated
        gene.mutate = MagicMock()  # ensure it's not used

        with patch(
            "app.services.auto_strategy.core.engine.ga_utils.GeneticUtils.extract_gene_params",
            return_value={"id": "mutated-id"},
        ):
            wrapper = create_deap_mutate_wrapper(StrategyGene, population, config)
            result = wrapper(gene)

        assert isinstance(result, tuple)
        assert len(result) == 1
        gene.adaptive_mutate.assert_called_once()
        # Verify population and config were passed
        args, _ = gene.adaptive_mutate.call_args
        assert args[0] is population
        assert args[1] is config
        gene.mutate.assert_not_called()

    def test_adaptive_mutate_uses_base_mutation_rate(self) -> None:
        config = SimpleNamespace(mutation_rate=0.07)
        gene = _make_gene_with_methods()
        mutated = _make_gene_with_methods()
        gene.adaptive_mutate.return_value = mutated
        population = [_make_gene_with_methods()]

        with patch(
            "app.services.auto_strategy.core.engine.ga_utils.GeneticUtils.extract_gene_params",
            return_value={},
        ):
            wrapper = create_deap_mutate_wrapper(StrategyGene, population, config)
            wrapper(gene)

        # base_mutation_rate keyword is set to config.mutation_rate
        _, kwargs = gene.adaptive_mutate.call_args
        assert kwargs.get("base_mutation_rate") == 0.07

    def test_wrapper_constructs_new_individual_from_extracted_params(self) -> None:
        config = SimpleNamespace(mutation_rate=0.1)
        gene = _make_gene_with_methods()
        mutated = _make_gene_with_methods()
        gene.mutate.return_value = mutated

        # Track constructor calls
        instances: list = []

        class FakeIndividual:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                instances.append(self)

        with patch(
            "app.services.auto_strategy.core.engine.ga_utils.GeneticUtils.extract_gene_params",
            return_value={"id": "extracted-id", "x": 1},
        ):
            wrapper = create_deap_mutate_wrapper(FakeIndividual, None, config)
            result = wrapper(gene)

        # The wrapper creates a FakeIndividual from extracted params
        assert isinstance(result, tuple)
        assert len(instances) == 1
        assert instances[0].kwargs == {"id": "extracted-id", "x": 1}

    def test_wrapper_returns_tuple_of_one_individual(self) -> None:
        """DEAPの慣例として、operatorは1要素タプルを返す"""
        config = SimpleNamespace(mutation_rate=0.1)
        gene = _make_gene_with_methods()
        gene.mutate.return_value = gene

        with patch(
            "app.services.auto_strategy.core.engine.ga_utils.GeneticUtils.extract_gene_params",
            return_value={},
        ):
            wrapper = create_deap_mutate_wrapper(StrategyGene, None, config)
            result = wrapper(gene)

        assert isinstance(result, tuple)
        assert len(result) == 1
