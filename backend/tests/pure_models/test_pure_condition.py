"""
Test for Condition model
"""
import pytest
from backend.app.services.auto_strategy.models.condition import Condition

class TestCondition:
    def test_init_with_str_operands(self):
        condition = Condition(left_operand="close", operator=">", right_operand="open")

        assert condition.left_operand == "close"
        assert condition.operator == ">"
        assert condition.right_operand == "open"

    def test_init_with_float_operands(self):
        condition = Condition(left_operand=1.5, operator="=", right_operand=2.0)

        assert condition.left_operand == 1.5
        assert condition.operator == "="
        assert condition.right_operand == 2.0

    def test_post_init_normalize_int_to_float(self):
        condition = Condition(left_operand=1, operator=">", right_operand=2)

        assert isinstance(condition.left_operand, float)
        assert isinstance(condition.right_operand, float)
        assert condition.left_operand == 1.0
        assert condition.right_operand == 2.0

    def test_post_init_ignores_dict(self):
        condition = Condition(left_operand={"type": "close"}, operator=">", right_operand="open")

        assert condition.left_operand == {"type": "close"}
        assert condition.right_operand == "open"

    def test_post_init_handles_non_int_non_dict_operands(self):
        """文字列や他の型は変換しないことを確認"""
        condition = Condition(left_operand="123", operator=">", right_operand="open")  # 文字列オペランド
        assert condition.left_operand == "123"  # 変換されない

    def test_post_init_only_converts_int_operands(self):
        """intのみ変換され、他の数値型は変換されない"""
        condition = Condition(left_operand=1.5, operator=">", right_operand=2)  # 右オペランドはint
        assert condition.left_operand == 1.5  # floatのまま
        assert isinstance(condition.right_operand, float)  # intがfloatに変換

    def test_validate_with_valid_condition(self):
        """有効な条件のバリデーション"""
        condition = Condition(left_operand="close", operator=">", right_operand="open")
        assert condition.validate() is True

    def test_validate_with_invalid_operator(self):
        """無効な演算子のバリデーション"""
        condition = Condition(left_operand="close", operator="invalid", right_operand="open")
        assert condition.validate() is False

    def test_validate_with_none_operand(self):
        """Noneオペランドのバリデーション"""
        condition = Condition(left_operand=None, operator=">", right_operand="open")
        assert condition.validate() is False

    def test_validate_with_empty_string_operand(self):
        """空文字オペランドのバリデーション"""
        condition = Condition(left_operand="", operator=">", right_operand="open")
        assert condition.validate() is False

    def test_validate_with_invalid_dict_operand(self):
        """無効な辞書オペランドのバリデーション"""
        condition = Condition(left_operand={"invalid": "key"}, operator=">", right_operand="open")
        assert condition.validate() is False