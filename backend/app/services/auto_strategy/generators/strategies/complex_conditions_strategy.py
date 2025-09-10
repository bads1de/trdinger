"""
ComplexConditions strategy for condition generation.
"""

import logging
import random
from typing import List, Union
from .base_strategy import ConditionStrategy
from ...models.strategy_models import IndicatorGene, Condition
from ...constants import IndicatorType

logger = logging.getLogger(__name__)

class ComplexConditionsStrategy(ConditionStrategy):
    """Strategy for generating complex conditions with multiple indicator combination."""

    def generate_conditions(
        self, indicators: List[IndicatorGene]
    ):
        """
        Generate conditions by combining multiple indicators.

        Uses up to 3 indicators to create more complex condition sets.
        """
        long_conditions = []
        short_conditions = []

        # Use up to 3 indicators to create balanced conditions
        selected_indicators = indicators[:3]

        for indicator in selected_indicators:
            if not indicator.enabled:
                continue

            # Get indicator type to determine which condition creation method to use
            indicator_type = self.condition_generator._get_indicator_type(indicator)

            try:
                # Use the appropriate condition creation method based on indicator type
                if indicator_type == IndicatorType.MOMENTUM:
                    long_conds = self.condition_generator._create_momentum_long_conditions(indicator)
                    short_conds = self.condition_generator._create_momentum_short_conditions(indicator)
                elif indicator_type == IndicatorType.TREND:
                    long_conds = self.condition_generator._create_trend_long_conditions(indicator)
                    short_conds = self.condition_generator._create_trend_short_conditions(indicator)
                else:
                    # Unknown indicator type - use generic conditions
                    long_conds = self.condition_generator._generic_long_conditions(indicator)
                    short_conds = self.condition_generator._generic_short_conditions(indicator)

                if long_conds:
                    long_conditions.extend(long_conds)

                # Ensure short conditions are also generated for each indicator
                if short_conds:
                    short_conditions.extend(short_conds)

            except Exception as e:
                self.condition_generator.logger.warning(f"Error generating conditions for {indicator.type}: {e}")
                # Fallback to generic conditions
                long_conditions.extend(self.condition_generator._generic_long_conditions(indicator))
                short_conditions.extend(self.condition_generator._generic_short_conditions(indicator))

        # If no conditions were generated, try with more indicators
        if not long_conditions:
            for indicator in indicators[:2]:
                if not indicator.enabled:
                    continue
                long_conditions.extend(self.condition_generator._generic_long_conditions(indicator))
                short_conditions.extend(self.condition_generator._generic_short_conditions(indicator))

                # One more attempt with fallback conditions
                if not long_conditions:
                    return self.condition_generator._generate_fallback_conditions()

        # Convert to appropriate return types
        long_result = list(long_conditions)
        short_result = list(short_conditions)
        exit_result = []

        return long_result, short_result, exit_result