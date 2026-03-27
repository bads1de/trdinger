"""
EvaluatorWrapper のユニットテスト

並列処理で使用する評価関数のラッパークラスをテストします。
"""

from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.core.evaluation.evaluator_wrapper import (
    EvaluatorWrapper,
)
from app.services.auto_strategy.core.evaluation.individual_evaluator import (
    IndividualEvaluator,
)


@pytest.fixture
def mock_evaluator():
    """モックのIndividualEvaluatorを作成"""
    evaluator = MagicMock(spec=IndividualEvaluator)
    evaluator.evaluate.return_value = (0.75,)
    return evaluator


@pytest.fixture
def ga_config():
    """GAConfigを作成"""
    return GAConfig(
        population_size=20,
        generations=10,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elite_size=2,
    )


class TestEvaluatorWrapper:
    """EvaluatorWrapperのテストクラス"""

    def test_init(self, mock_evaluator, ga_config):
        """初期化が正しく行われること"""
        wrapper = EvaluatorWrapper(mock_evaluator, ga_config)
        assert wrapper.evaluator is mock_evaluator
        assert wrapper.config is ga_config

    def test_call_returns_fitness(self, mock_evaluator, ga_config):
        """呼び出し時にフィットネス値が返されること"""
        wrapper = EvaluatorWrapper(mock_evaluator, ga_config)
        mock_individual = MagicMock()

        result = wrapper(mock_individual)

        assert result == (0.75,)
        mock_evaluator.evaluate.assert_called_once_with(mock_individual, ga_config)

    def test_call_with_different_individuals(self, mock_evaluator, ga_config):
        """異なる個体でも正しく評価されること"""
        wrapper = EvaluatorWrapper(mock_evaluator, ga_config)

        individual1 = MagicMock()
        individual2 = MagicMock()

        result1 = wrapper(individual1)
        result2 = wrapper(individual2)

        assert result1 == (0.75,)
        assert result2 == (0.75,)
        assert mock_evaluator.evaluate.call_count == 2

    def test_call_delegates_to_evaluator(self, ga_config):
        """評価がevaluatorに委譲されること"""
        mock_evaluator = MagicMock(spec=IndividualEvaluator)
        mock_evaluator.evaluate.return_value = (0.5, 0.3)

        wrapper = EvaluatorWrapper(mock_evaluator, ga_config)
        mock_individual = MagicMock()

        result = wrapper(mock_individual)

        assert result == (0.5, 0.3)
        mock_evaluator.evaluate.assert_called_once_with(mock_individual, ga_config)

    def test_evaluator_error_propagates(self, mock_evaluator, ga_config):
        """evaluatorでエラーが発生した場合、エラーが伝播すること"""
        mock_evaluator.evaluate.side_effect = ValueError("Evaluation error")

        wrapper = EvaluatorWrapper(mock_evaluator, ga_config)
        mock_individual = MagicMock()

        with pytest.raises(ValueError, match="Evaluation error"):
            wrapper(mock_individual)
