from unittest.mock import MagicMock
from app.services.auto_strategy.generators.strategies.complex_conditions_strategy import (
    ComplexConditionsStrategy,
)
from app.services.auto_strategy.models.condition import Condition, ConditionGroup
from app.services.auto_strategy.models.indicator_gene import IndicatorGene


class MockConditionGenerator:
    def __init__(self):
        self.logger = MagicMock()

    def _get_indicator_type(self, ind):
        return "UNKNOWN"

    def _generic_long_conditions(self, ind):
        return [Condition(ind.type, ">", 50)]

    def _generic_short_conditions(self, ind):
        return [Condition(ind.type, "<", 50)]

    def _create_momentum_long_conditions(self, ind):
        return [Condition(ind.type, ">", 50)]

    def _create_momentum_short_conditions(self, ind):
        return [Condition(ind.type, "<", 50)]

    def _create_trend_long_conditions(self, ind):
        return [Condition(ind.type, ">", 50)]

    def _create_trend_short_conditions(self, ind):
        return [Condition(ind.type, "<", 50)]


def test_generate_hierarchical_structure():
    generator = MockConditionGenerator()
    strategy = ComplexConditionsStrategy(generator)

    indicators = [
        IndicatorGene(type="rsi", parameters={}, enabled=True),
        IndicatorGene(type="macd", parameters={}, enabled=True),
        IndicatorGene(type="bb", parameters={}, enabled=True),
        IndicatorGene(type="sma", parameters={}, enabled=True),
    ]

    # We expect some hierarchy (ConditionGroup)
    found_group = False

    # Try multiple times if randomness is involved
    for _ in range(5):
        longs, shorts, exits = strategy.generate_conditions(indicators)
        for cond in longs + shorts:
            if isinstance(cond, ConditionGroup):
                # We want to ensure it's not just a wrapper but contains actual structure
                if len(cond.conditions) >= 2:
                    found_group = True
                    break
        if found_group:
            break

    assert (
        found_group
    ), "ComplexConditionsStrategy should generate ConditionGroups (hierarchical structure)"
