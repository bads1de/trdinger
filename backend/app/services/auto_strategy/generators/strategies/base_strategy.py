"""
Base strategy classes for condition generation.
"""
from abc import ABC, abstractmethod
import random
from typing import List, Tuple, TypeAlias
from ...models.strategy_models import IndicatorGene, Condition, ConditionGroup, Condition
from ...constants import IndicatorType, StrategyType

ConditionList: TypeAlias = List[Condition]

class ConditionStrategy(ABC):
    """Base class for condition generation strategies."""

    def __init__(self, condition_generator):
        """
        Initialize strategy with context.

        Args:
            condition_generator: Reference to ConditionGenerator for shared state
        """
        self.condition_generator = condition_generator

    @abstractmethod
    def generate_conditions(
        self, indicators: List[IndicatorGene]
    ) -> Tuple[List, List, List]:
        """
        Generate conditions for the strategy.

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)
        """
        pass

    # Helper methods that can be used by subclasses
    def _classify_indicators_by_type(
        self, indicators: List[IndicatorGene]
    ) -> dict:
        """
        Classify indicators by their type using the condition generator's classification.

        Args:
            indicators: List of indicators to classify

        Returns:
            Dictionary mapping IndicatorType to list of indicators
        """
        return self.condition_generator._dynamic_classify(indicators)

    def _create_generic_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Create generic long conditions for an indicator."""
        return self.condition_generator._generic_long_conditions(indicator)

    def _create_generic_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Create generic short conditions for an indicator."""
        return self.condition_generator._generic_short_conditions(indicator)

    def _create_ml_long_conditions__(
        self, indicators: List[IndicatorGene]
    ) -> List[Condition]:
        """Create ML-based long conditions."""
        return self.condition_generator._create_ml_long_conditions(indicators)