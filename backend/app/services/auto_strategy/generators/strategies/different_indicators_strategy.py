"""
DifferentIndicators strategy for condition generation.
"""

import logging
import random
from typing import List
from .base_strategy import ConditionStrategy
from ...models.strategy_models import IndicatorGene, Condition
from ...constants import IndicatorType

logger = logging.getLogger(__name__)


class DifferentIndicatorsStrategy(ConditionStrategy):
    """Strategy for generating conditions when different types of indicators are available."""

    def generate_conditions(self, indicators: List[IndicatorGene]):
        """
        Generate conditions using different indicator types.

        Combines trend + momentum or momentum + volatility indicators.
        """
        logger.debug(
            f"Generating conditions for {len(indicators)} indicators using DifferentIndicators strategy"
        )

        # Group indicators by type
        indicators_by_type = self._classify_indicators_by_type(indicators)
        logger.debug(
            f"Indicator classification: {[f'{k.name}:{len(v)}' for k, v in indicators_by_type.items() if v]}"
        )

        long_conditions = []
        short_conditions = []

        # ML indicators get priority
        ml_indicators = [
            ind for ind in indicators if ind.enabled and ind.type.startswith("ML_")
        ]
        logger.debug(f"ML indicators count: {len(ml_indicators)}")

        # Add trend-based conditions for long
        if indicators_by_type.get(IndicatorType.TREND):
            selected_trend = self._create_trend_long_conditions(
                random.choice(indicators_by_type[IndicatorType.TREND])
            )
            long_conditions.extend(selected_trend)
            logger.debug(f"Added {len(selected_trend)} trend long conditions")

        # Add momentum conditions for long
        if indicators_by_type.get(IndicatorType.MOMENTUM):
            selected_momentum = self._create_momentum_long_conditions(
                random.choice(indicators_by_type[IndicatorType.MOMENTUM])
            )
            long_conditions.extend(selected_momentum)
            logger.debug(f"Added {len(selected_momentum)} momentum long conditions")

        # Add ML conditions for long
        if ml_indicators:
            ml_long_conditions = self.condition_generator._create_ml_long_conditions(
                ml_indicators
            )
            long_conditions.extend(ml_long_conditions)
            logger.debug(f"Added {len(ml_long_conditions)} ML long conditions")

        # Short conditions (opposite direction)
        if indicators_by_type.get(IndicatorType.TREND):
            selected_trend_short = self._create_trend_short_conditions(
                random.choice(indicators_by_type[IndicatorType.TREND])
            )
            short_conditions.extend(selected_trend_short)
            logger.debug(f"Added {len(selected_trend_short)} trend short conditions")

        if indicators_by_type.get(IndicatorType.MOMENTUM):
            selected_momentum_short = self._create_momentum_short_conditions(
                random.choice(indicators_by_type[IndicatorType.MOMENTUM])
            )
            short_conditions.extend(selected_momentum_short)
            logger.debug(
                f"Added {len(selected_momentum_short)} momentum short conditions"
            )

        # ML opposite signals for short
        if ml_indicators and len(ml_indicators) >= 2:
            if any(ind.type == "ML_DOWN_PROB" for ind in ml_indicators):
                short_conditions.append(
                    Condition(
                        left_operand="ML_DOWN_PROB", operator=">", right_operand=0.6
                    )
                )

        # Ensure we have at least basic conditions
        if not long_conditions:
            long_conditions = [
                Condition(left_operand="close", operator=">", right_operand="open")
            ]

        if not short_conditions:
            short_conditions = [
                Condition(left_operand="close", operator="<", right_operand="open")
            ]

        # Convert to appropriate types
        long_result = list(long_conditions)
        short_result = list(short_conditions)
        exit_result = []

        logger.debug(
            f"Generated {len(long_result)} long and {len(short_result)} short conditions"
        )
        return long_result, short_result, exit_result

    def _create_trend_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Create trend-based long conditions."""
        return self.condition_generator._create_trend_long_conditions(indicator)

    def _create_trend_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Create trend-based short conditions."""
        return self.condition_generator._create_trend_short_conditions(indicator)

    def _create_momentum_long_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Create momentum-based long conditions."""
        return self.condition_generator._create_momentum_long_conditions(indicator)

    def _create_momentum_short_conditions(
        self, indicator: IndicatorGene
    ) -> List[Condition]:
        """Create momentum-based short conditions."""
        return self.condition_generator._create_momentum_short_conditions(indicator)
