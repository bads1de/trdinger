"""
Test for ConditionGroup model
"""
import pytest
from unittest.mock import patch
from backend.app.services.auto_strategy.models.condition import ConditionGroup, Condition

class TestConditionGroup:
    def test_init_empty(self):
        group = ConditionGroup()

        assert group.conditions == []
        assert group.is_empty() is True

    def test_init_with_conditions(self):
        conditions = [Condition(left_operand="a", operator=">", right_operand="b")]

        group = ConditionGroup(conditions=conditions)

        assert group.conditions == conditions
        assert group.is_empty() is False

    def test_is_empty_with_conditions(self):
        conditions = [Condition(left_operand="a", operator=">", right_operand="b")]

        group = ConditionGroup(conditions=conditions)

        assert group.is_empty() is False

    def test_is_empty_after_clear(self):
        conditions = [Condition(left_operand="a", operator=">", right_operand="b")]

        group = ConditionGroup(conditions=conditions)

        group.conditions = []

        assert group.is_empty() is True
    def test_validate_with_valid_conditions(self):
        condition1 = Condition(left_operand="close", operator=">", right_operand="open")
        condition2 = Condition(left_operand=1.0, operator="<", right_operand=2.0)
        group = ConditionGroup(conditions=[condition1, condition2])

        assert group.validate() is True

    def test_validate_with_invalid_conditions(self):
        valid_condition = Condition(left_operand="close", operator=">", right_operand="open")
        invalid_condition = Condition(left_operand="close", operator="invalid", right_operand="open")
        group = ConditionGroup(conditions=[valid_condition, invalid_condition])

        assert group.validate() is False

    def test_validate_empty_group(self):
        group = ConditionGroup()

        assert group.validate() is True

    @patch('backend.app.services.auto_strategy.models.validator.GeneValidator')
    def test_validate_with_mock_validator(self, mock_validator_cls):
        """モックしたバリデーターでvalidateメソッドテスト"""
        mock_validator = mock_validator_cls.return_value
        # 1つ目の条件はvalid、2つ目はinvalid
        mock_validator.validate_condition.side_effect = [(True, ""), (False, "invalid operator")]

        valid_condition = Condition(left_operand="close", operator=">", right_operand="open")
        invalid_condition = Condition(left_operand="close", operator="invalid", right_operand="open")

        group = ConditionGroup(conditions=[valid_condition, invalid_condition])

        assert group.validate() is False

        # validate_conditionが2回呼ばれたことを確認
        assert mock_validator.validate_condition.call_count == 2

    @patch('backend.app.services.auto_strategy.models.validator.GeneValidator')
    def test_validate_all_valid_conditions(self, mock_validator_cls):
        """全ての条件が有効な場合"""
        mock_validator = mock_validator_cls.return_value
        mock_validator.validate_condition.return_value = (True, "")

        condition1 = Condition(left_operand="close", operator=">", right_operand="open")
        condition2 = Condition(left_operand="high", operator="<", right_operand="low")

        group = ConditionGroup(conditions=[condition1, condition2])

        assert group.validate() is True