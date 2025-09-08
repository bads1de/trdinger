import pytest
from backend.app.services.auto_strategy.models.pure_strategy_models import PureCondition

class TestPureCondition:
    def test_init_with_str_operands(self):
        condition = PureCondition(left_operand="close", operator=">", right_operand="open")

        assert condition.left_operand == "close"
        assert condition.operator == ">"
        assert condition.right_operand == "open"

    def test_init_with_float_operands(self):
        condition = PureCondition(left_operand=1.5, operator="=", right_operand=2.0)

        assert condition.left_operand == 1.5
        assert condition.operator == "="
        assert condition.right_operand == 2.0

    def test_post_init_normalize_int_to_float(self):
        condition = PureCondition(left_operand=1, operator=">", right_operand=2)

        assert isinstance(condition.left_operand, float)
        assert isinstance(condition.right_operand, float)
        assert condition.left_operand == 1.0
        assert condition.right_operand == 2.0

    def test_post_init_ignores_dict(self):
        condition = PureCondition(left_operand={"type": "close"}, operator=">", right_operand="open")

        assert condition.left_operand == {"type": "close"}
        assert condition.right_operand == "open"