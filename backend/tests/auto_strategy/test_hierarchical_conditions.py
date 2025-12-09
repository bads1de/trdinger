import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.services.auto_strategy.models.condition import Condition, ConditionGroup
from app.services.auto_strategy.core.condition_evaluator import ConditionEvaluator
from app.services.auto_strategy.models.validator import GeneValidator
from app.services.auto_strategy.models.strategy_gene import StrategyGene
from app.services.auto_strategy.models.indicator_gene import IndicatorGene


class MockStrategy:
    def __init__(self, data_dict):
        for k, v in data_dict.items():
            setattr(self, k, pd.Series(v))
        self.data = MagicMock()  # For OHLCV checks if needed


def test_hierarchical_condition_evaluation():
    # Goal: (A > 10 AND B < 5) OR C > 100
    # Data: A=20 (True), B=4 (True) -> AND is True
    #       C=50 (False)
    #       Result -> True OR False -> True

    cond_a = Condition("A", ">", 10)
    cond_b = Condition("B", "<", 5)
    cond_c = Condition("C", ">", 100)

    # We expect ConditionGroup to support 'operator' and nested 'conditions'
    try:
        group_and = ConditionGroup(operator="AND", conditions=[cond_a, cond_b])
        group_or = ConditionGroup(operator="OR", conditions=[group_and, cond_c])
    except TypeError:
        pytest.fail("ConditionGroup does not support 'operator' argument")

    strategy = MockStrategy({"A": [20.0], "B": [4.0], "C": [50.0]})

    evaluator = ConditionEvaluator()
    # evaluate_conditions takes a list of conditions (implicit AND).
    # Passing [group_or] means "Evaluate group_or".
    result = evaluator.evaluate_conditions([group_or], strategy)

    assert result is True


def test_hierarchical_condition_evaluation_complex_false():
    # Goal: (A > 10 AND B < 5) OR C > 100
    # Data: A=20 (True), B=6 (False) -> AND is False
    #       C=50 (False)
    #       Result -> False OR False -> False

    cond_a = Condition("A", ">", 10)
    cond_b = Condition("B", "<", 5)
    cond_c = Condition("C", ">", 100)

    try:
        group_and = ConditionGroup(operator="AND", conditions=[cond_a, cond_b])
        group_or = ConditionGroup(operator="OR", conditions=[group_and, cond_c])
    except TypeError:
        pytest.fail("ConditionGroup does not support 'operator' argument")

    strategy = MockStrategy({"A": [20.0], "B": [6.0], "C": [50.0]})

    evaluator = ConditionEvaluator()
    result = evaluator.evaluate_conditions([group_or], strategy)

    assert result is False


def test_hierarchical_validation():
    """ネストされたConditionGroupのバリデーションテスト"""
    # Mock get_all_indicators to include our test operands "A", "B", "C"
    with patch(
        "app.services.auto_strategy.models.validator.get_all_indicators",
        return_value=["A", "B", "C"],
    ):
        validator = GeneValidator()

        # Valid Nested Structure: (A > 10 AND B < 5) OR C > 100
        cond_a = Condition("A", ">", 10)
        cond_b = Condition("B", "<", 5)
        cond_c = Condition("C", ">", 100)

        group_and = ConditionGroup(operator="AND", conditions=[cond_a, cond_b])
        group_or = ConditionGroup(operator="OR", conditions=[group_and, cond_c])

        strategy = StrategyGene(
            entry_conditions=[group_or],
            exit_conditions=[Condition("A", ">", 10)],
            indicators=[IndicatorGene(type="A", enabled=True, parameters={})],
        )

        # This should pass if validator handles recursion correctly
        is_valid, errors = validator.validate_strategy_gene(strategy)

        assert is_valid, f"Validation failed: {errors}"


def test_hierarchical_mutation_safety():
    """ConditionGroupが突然変異によって破壊されないことを確認"""
    cond_a = Condition("A", ">", 10)
    cond_b = Condition("B", "<", 5)

    # AND Group
    group = ConditionGroup(operator="AND", conditions=[cond_a, cond_b])

    strategy = StrategyGene(
        id="test",
        indicators=[],
        entry_conditions=[group],
        exit_conditions=[],
        risk_management={},
        metadata={},
    )

    from app.services.auto_strategy.core.genetic_operators import (
        mutate_strategy_gene_pure,
    )

    # Force mutation on conditions
    # We loop multiple times to ensure the random chance hits condition mutation logic
    # _mutate_conditions probability is mutation_rate * 0.5
    # Then inside, another 0.5 chance for entry_conditions.
    # So we need high mutation rate and iterations.

    for _ in range(30):
        # mutation_rate 1.0 ensures mutation attempt if logic allows
        mutated = mutate_strategy_gene_pure(strategy, mutation_rate=1.0)

        # Check if entry_conditions[0] is still a ConditionGroup and operator is valid
        # Mutated strategy creates a deep copy, so we check the mutated object.
        if mutated.entry_conditions:
            mutated_group = mutated.entry_conditions[0]
            if isinstance(mutated_group, ConditionGroup):
                assert mutated_group.operator in [
                    "AND",
                    "OR",
                ], f"Operator was corrupted to: {mutated_group.operator}"
