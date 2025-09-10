"""
IndicatorCharacteristics strategy for ML and specialized indicators.
"""

import logging
from typing import List, Union
import random
from .base_strategy import ConditionStrategy
from ...models.strategy_models import IndicatorGene, Condition
from ...constants import IndicatorType

logger = logging.getLogger(__name__)

class IndicatorCharacteristicsStrategy(ConditionStrategy):
    """Strategy for generating conditions based on indicator characteristics (primarily ML)."""

    def generate_conditions(
        self, indicators: List[IndicatorGene]
    ):
        """
        Generate conditions focusing on indicator characteristics.

        Primarily handles ML indicators and their probability-based conditions.
        """
        long_conditions = []
        short_conditions = []

        # Focus on ML indicators for this strategy
        ml_indicators = [
            ind for ind in indicators if ind.enabled and ind.type.startswith("ML_")
        ]

        if ml_indicators:
            # Generate ML-specific long conditions
            ml_long_conds = self.condition_generator._create_ml_long_conditions(ml_indicators)
            if ml_long_conds:
                long_conditions.extend(ml_long_conds)

            # Generate ML-specific short conditions
            # Look for complementary ML indicators for short signals
            if len(ml_indicators) >= 1:
                # Add short conditions based on ML probabilities
                short_conditions.extend(self._create_ml_short_conditions(ml_indicators))

        # If still no conditions and we have regular indicators, use limited generic conditions
        if not long_conditions:
            # For non-ML indicators, use very basic conditions
            regular_indicators = [
                ind for ind in indicators if ind.enabled and not ind.type.startswith("ML_")
            ]

            if regular_indicators:
                # Only use first indicator for this strategy to keep it focused
                first_indicator = regular_indicators[0]
                long_conditions.extend(self.condition_generator._generic_long_conditions(first_indicator))
                short_conditions.extend(self.condition_generator._generic_short_conditions(first_indicator))

        # Ensure we have at least basic conditions
        if not long_conditions or not short_conditions:
            # Use fallback conditions as last resort
            try:
                fallback_result = self.condition_generator._generate_fallback_conditions()
                # Unpack only if it's a tuple
                if isinstance(fallback_result, tuple) and len(fallback_result) == 3:
                    longfallback, shortfallback, _ = fallback_result
                    if not long_conditions:
                        long_conditions = longfallback
                    if not short_conditions:
                        short_conditions = shortfallback
                else:
                    # If not a proper tuple, use default fallback
                    if not long_conditions:
                        long_conditions = [Condition(left_operand="close", operator=">", right_operand="open")]
                    if not short_conditions:
                        short_conditions = [Condition(left_operand="close", operator="<", right_operand="open")]
                    logger.warning("Invalid fallback result format, using default conditions")
            except Exception as e:
                logger.error(f"Error in fallback generation: {e}")
                # Use absolute default conditions
                if not long_conditions:
                    long_conditions = [Condition(left_operand="close", operator=">", right_operand="open")]
                if not short_conditions:
                    short_conditions = [Condition(left_operand="close", operator="<", right_operand="open")]

        # Convert to appropriate return types
        long_result = list(long_conditions)
        short_result = list(short_conditions)
        exit_result = []

        return long_result, short_result, exit_result

    def _create_ml_short_conditions(self, ml_indicators: List[IndicatorGene]) -> List[Condition]:
        """
        Create ML-based short conditions.

        Args:
            ml_indicators: List of ML indicators

        Returns:
            List of short entry conditions
        """
        short_conditions = []

        # Check for specific ML probability indicators
        for indicator in ml_indicators:
            if indicator.type == "ML_DOWN_PROB":
                short_conditions.append(
                    Condition(
                        left_operand="ML_DOWN_PROB",
                        operator=">",
                        right_operand=0.6,
                    )
                )
            elif indicator.type == "ML_UP_PROB":
                # For upward probability, we can use it inversely for short
                short_conditions.append(
                    Condition(
                        left_operand="ML_UP_PROB",
                        operator="<",
                        right_operand=0.3,
                    )
                )

        # If we don't have specific indicators, use range probability
        if not short_conditions:
            range_indicators = [ind for ind in ml_indicators if ind.type == "ML_RANGE_PROB"]
            if range_indicators:
                short_conditions.append(
                    Condition(
                        left_operand="ML_RANGE_PROB",
                        operator=">",
                        right_operand=0.7,
                    )
                )

        return short_conditions