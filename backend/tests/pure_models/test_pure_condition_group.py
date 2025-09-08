import pytest
from backend.app.services.auto_strategy.models.pure_strategy_models import PureConditionGroup, PureCondition

class TestPureConditionGroup:
    def test_init_empty(self):
        group = PureConditionGroup()

        assert group.conditions == []
        assert group.is_empty() is True

    def test_init_with_conditions(self):
        conditions = [PureCondition(left_operand="a", operator=">", right_operand="b")]

        group = PureConditionGroup(conditions=conditions)

        assert group.conditions == conditions
        assert group.is_empty() is False

    def test_is_empty_with_conditions(self):
        conditions = [PureCondition(left_operand="a", operator=">", right_operand="b")]

        group = PureConditionGroup(conditions=conditions)

        assert group.is_empty() is False

    def test_is_empty_after_clear(self):
        conditions = [PureCondition(left_operand="a", operator=">", right_operand="b")]

        group = PureConditionGroup(conditions=conditions)

        group.conditions = []

        assert group.is_empty() is True