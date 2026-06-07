"""
ConditionValidator のテスト

``app.services.auto_strategy.genes.validators.condition_validator.ConditionValidator`` の
``validate_condition``, ``_is_valid_operand_detailed``, ``_is_indicator_name``,
``clean_condition``, ``_extract_operand_from_dict``, ``_is_trivial_condition`` を検証します。
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from app.services.auto_strategy.genes.conditions import Condition, ConditionGroup
from app.services.auto_strategy.genes.validators.condition_validator import (
    ConditionValidator,
)


@pytest.fixture
def indicator_validator():
    """シンプルな indicator_validator モック"""
    mock = MagicMock()
    mock.valid_indicator_types = ["SMA", "RSI", "EMA", "MACD"]
    return mock


@pytest.fixture
def validator(indicator_validator):
    return ConditionValidator(indicator_validator=indicator_validator)


def _make_condition(left="SMA_20", op=">", right="SMA_50") -> Mock:
    c = Mock()
    c.operator = op
    c.left_operand = left
    c.right_operand = right
    return c


class TestInit:
    def test_stores_indicator_validator(self, indicator_validator):
        v = ConditionValidator(indicator_validator=indicator_validator)
        assert v.indicator_validator is indicator_validator
        assert v.valid_operators is not None
        assert v.valid_data_sources is not None


class TestValidateCondition:
    def test_returns_false_when_attrs_missing(self, validator):
        c = Mock(spec=[])  # no attributes
        ok, msg = validator.validate_condition(c)
        assert ok is False
        assert "属性がありません" in msg

    def test_returns_false_for_invalid_operator(self, validator):
        c = _make_condition(op="?")
        ok, msg = validator.validate_condition(c)
        assert ok is False
        assert "演算子" in msg

    def test_returns_false_for_invalid_left_operand(self, validator):
        c = _make_condition(left="invalid")
        ok, msg = validator.validate_condition(c)
        assert ok is False
        assert "左オペランド" in msg

    def test_returns_false_for_invalid_right_operand(self, validator):
        c = _make_condition(right="@bad")
        ok, msg = validator.validate_condition(c)
        assert ok is False
        assert "右オペランド" in msg

    def test_returns_false_for_trivial(self, validator):
        c = _make_condition(left="close", right="close")
        ok, msg = validator.validate_condition(c)
        assert ok is False
        assert "シンプル" in msg

    def test_returns_true_for_valid(self, validator):
        c = _make_condition(left="SMA_20", right=10.0)
        ok, msg = validator.validate_condition(c)
        assert ok is True
        assert msg == ""


class TestIsValidOperandDetailed:
    def test_returns_false_for_none(self, validator):
        ok, msg = validator._is_valid_operand_detailed(None)
        assert ok is False
        assert "None" in msg

    def test_returns_true_for_int(self, validator):
        ok, _ = validator._is_valid_operand_detailed(42)
        assert ok is True

    def test_returns_true_for_float(self, validator):
        ok, _ = validator._is_valid_operand_detailed(3.14)
        assert ok is True

    def test_returns_true_for_numeric_string(self, validator):
        ok, _ = validator._is_valid_operand_detailed("3.14")
        assert ok is True

    def test_returns_true_for_data_source_string(self, validator):
        ok, _ = validator._is_valid_operand_detailed("close")
        assert ok is True

    def test_returns_true_for_indicator_name_string(self, validator):
        ok, _ = validator._is_valid_operand_detailed("SMA_20")
        assert ok is True

    def test_returns_false_for_empty_string(self, validator):
        ok, msg = validator._is_valid_operand_detailed("   ")
        assert ok is False
        assert "空" in msg

    def test_returns_false_for_invalid_string(self, validator):
        ok, msg = validator._is_valid_operand_detailed("not_valid_xyz")
        assert ok is False
        assert "無効な文字列" in msg

    def test_dict_indicator_valid(self, validator):
        op = {"type": "indicator", "name": "SMA_20"}
        ok, _ = validator._is_valid_operand_detailed(op)
        assert ok is True

    def test_dict_indicator_invalid_name(self, validator):
        op = {"type": "indicator", "name": "BOGUS"}
        ok, msg = validator._is_valid_operand_detailed(op)
        assert ok is False
        assert "指標名" in msg

    def test_dict_price_valid(self, validator):
        op = {"type": "price", "name": "close"}
        ok, _ = validator._is_valid_operand_detailed(op)
        assert ok is True

    def test_dict_price_invalid(self, validator):
        op = {"type": "price", "name": "invalid_source"}
        ok, msg = validator._is_valid_operand_detailed(op)
        assert ok is False
        assert "データソース" in msg

    def test_dict_value_valid(self, validator):
        op = {"type": "value", "value": 100.5}
        ok, _ = validator._is_valid_operand_detailed(op)
        assert ok is True

    def test_dict_value_invalid(self, validator):
        op = {"type": "value", "value": "not-a-number"}
        ok, msg = validator._is_valid_operand_detailed(op)
        assert ok is False
        assert "数値" in msg

    def test_dict_value_none(self, validator):
        op = {"type": "value", "value": None}
        ok, _ = validator._is_valid_operand_detailed(op)
        assert ok is True

    def test_dict_unknown_type(self, validator):
        op = {"type": "unknown", "name": "X"}
        ok, msg = validator._is_valid_operand_detailed(op)
        assert ok is False
        assert "辞書タイプ" in msg

    def test_unsupported_type(self, validator):
        op = [1, 2, 3]  # list
        ok, msg = validator._is_valid_operand_detailed(op)
        assert ok is False
        assert "未対応の型" in msg


class TestIsIndicatorName:
    def test_returns_false_for_empty(self, validator):
        assert validator._is_indicator_name("") is False
        assert validator._is_indicator_name(None) is False

    def test_returns_true_for_exact_match(self, validator):
        assert validator._is_indicator_name("SMA") is True

    def test_returns_true_for_underscore_split(self, validator):
        # SMA_20 → SMA 完全一致
        assert validator._is_indicator_name("SMA_20") is True

    def test_returns_true_for_uppercase_match(self, validator):
        # 大文字小文字無視
        assert validator._is_indicator_name("sma") is True
        assert validator._is_indicator_name("Sma_20") is True

    def test_returns_true_for_allowed_prefix(self, validator):
        # MACD は indicator type だが接頭辞 MACDS_xxx も許可
        assert validator._is_indicator_name("MACDS_signal") is True
        assert validator._is_indicator_name("ATR_14") is True
        assert validator._is_indicator_name("RSI_value") is True

    def test_returns_false_for_unknown(self, validator):
        assert validator._is_indicator_name("ZZZ_99") is False


class TestCleanCondition:
    def test_returns_true_for_condition_group(self, validator):
        g = ConditionGroup(conditions=[], operator="AND")
        assert validator.clean_condition(g) is True

    def test_strips_string_operands(self, validator):
        c = Condition(left_operand="  SMA_20  ", operator=">", right_operand="  10.5  ")
        validator.clean_condition(c)
        assert c.left_operand == "SMA_20"
        assert c.right_operand == "10.5"

    def test_converts_above_below_operators(self, validator):
        c = Condition(left_operand="SMA_20", operator="above", right_operand="SMA_50")
        validator.clean_condition(c)
        assert c.operator == ">"

        c2 = Condition(left_operand="SMA_20", operator="below", right_operand="SMA_50")
        validator.clean_condition(c2)
        assert c2.operator == "<"

    def test_handles_dict_operands(self, validator):
        c = Condition(
            left_operand={"type": "indicator", "name": "SMA"},
            operator=">",
            right_operand={"type": "value", "value": 10.0},
        )
        result = validator.clean_condition(c)
        assert result is True
        # dict から string 抽出されているはず
        assert c.left_operand == "SMA"
        assert c.right_operand == "10.0"

    def test_no_string_operand(self, validator):
        c = Mock()
        c.left_operand = 10.0
        c.right_operand = 20.0
        c.operator = ">"
        result = validator.clean_condition(c)
        assert result is True


class TestExtractOperandFromDict:
    def test_extract_indicator(self, validator):
        result = validator._extract_operand_from_dict(
            {"type": "indicator", "name": "RSI"}
        )
        assert result == "RSI"

    def test_extract_price(self, validator):
        result = validator._extract_operand_from_dict(
            {"type": "price", "name": "close"}
        )
        assert result == "close"

    def test_extract_value(self, validator):
        result = validator._extract_operand_from_dict({"type": "value", "value": 100.5})
        assert result == "100.5"

    def test_extract_value_none(self, validator):
        result = validator._extract_operand_from_dict({"type": "value", "value": None})
        assert result == ""

    def test_extract_unknown_type_falls_back_to_name(self, validator):
        result = validator._extract_operand_from_dict(
            {"type": "unknown", "name": "fallback_name"}
        )
        assert result == "fallback_name"


class TestIsTrivialCondition:
    def test_returns_false_when_attrs_missing(self, validator):
        c = Mock(spec=[])
        assert validator._is_trivial_condition(c) is False

    def test_returns_true_for_same_price_field(self, validator):
        c = _make_condition(left="close", right="close", op=">")
        assert validator._is_trivial_condition(c) is True

    def test_returns_true_for_trivial_numeric(self, validator):
        # right == 1.0 で価格フィールドとの比較
        c = _make_condition(left="close", right=1.0, op=">")
        assert validator._is_trivial_condition(c) is True

    def test_returns_true_for_trivial_zero(self, validator):
        c = _make_condition(left="close", right=0.0, op="<")
        assert validator._is_trivial_condition(c) is True

    def test_returns_true_for_large_value(self, validator):
        # |right| > 10 で価格比較
        c = _make_condition(left="close", right=100.0, op=">")
        assert validator._is_trivial_condition(c) is True

    def test_returns_false_for_non_trivial(self, validator):
        # 価格 × 価格以外
        c = _make_condition(left="SMA_20", right=2.0, op=">")
        assert validator._is_trivial_condition(c) is False

    def test_returns_false_for_normal_numeric(self, validator):
        # 価格とちょうどいい値
        c = _make_condition(left="close", right=5.0, op=">")
        assert validator._is_trivial_condition(c) is False
