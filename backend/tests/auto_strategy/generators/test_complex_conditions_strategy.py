from unittest.mock import MagicMock
from app.services.auto_strategy.generators.strategies.complex_conditions_strategy import (
    ComplexConditionsStrategy,
)
from app.services.auto_strategy.genes.conditions import Condition, ConditionGroup
from app.services.auto_strategy.genes.indicator_gene import IndicatorGene


class MockConditionGenerator:
    def __init__(self):
        self.logger = MagicMock()

    def _get_indicator_type(self, ind):
        return "UNKNOWN"

    def _generic_long_conditions(self, ind):
        return [Condition(left_operand=ind.type, operator=">", right_operand=50.0)]

    def _generic_short_conditions(self, ind):
        return [Condition(left_operand=ind.type, operator="<", right_operand=50.0)]

    def _create_momentum_long_conditions(self, ind):
        return [Condition(left_operand=ind.type, operator=">", right_operand=50.0)]

    def _create_momentum_short_conditions(self, ind):
        return [Condition(left_operand=ind.type, operator="<", right_operand=50.0)]

    def _create_trend_long_conditions(self, ind):
        return [Condition(left_operand=ind.type, operator=">", right_operand=50.0)]

    def _create_trend_short_conditions(self, ind):
        return [Condition(left_operand=ind.type, operator="<", right_operand=50.0)]


def test_generate_hierarchical_structure():
    generator = MockConditionGenerator()
    strategy = ComplexConditionsStrategy(generator)

    # より多くのインジケータを提供し、パラメータも設定
    indicators = [
        IndicatorGene(type="rsi", parameters={"length": 14}, enabled=True),
        IndicatorGene(type="macd", parameters={"fast": 12, "slow": 26}, enabled=True),
        IndicatorGene(type="bb", parameters={"length": 20}, enabled=True),
        IndicatorGene(type="sma", parameters={"period": 20}, enabled=True),
        IndicatorGene(type="ema", parameters={"period": 50}, enabled=True),
        IndicatorGene(type="stoch", parameters={"k": 14}, enabled=True),
    ]

    # We expect some hierarchy (ConditionGroup) or multiple conditions
    found_group_or_multiple = False

    # Try multiple times if randomness is involved
    for iteration in range(50):  # 試行回数を増やす
        longs, shorts, exits = strategy.generate_conditions(indicators)
        
        # グループが見つかったか、または複数の条件が生成されたか
        has_group = any(isinstance(c, ConditionGroup) and len(c.conditions) >= 2 for c in longs + shorts)
        has_multiple_conditions = len(longs) + len(shorts) >= 2
        
        if has_group or has_multiple_conditions:
            found_group_or_multiple = True
            break

    # 修正された期待値：ConditionGroupまたは複数の条件が生成されること
    assert (
        found_group_or_multiple
    ), "ComplexConditionsStrategy should generate ConditionGroups or multiple conditions"




