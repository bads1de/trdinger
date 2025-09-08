import pytest
import sys
import os
from unittest.mock import patch

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

from app.services.auto_strategy.utils.validation_utils import ValidationUtils


class TestValidationUtils:
    """ValidationUtilsクラスのテスト"""

    def test_validate_range_normal(self):
        """範囲内の値はTrueを返す"""
        assert ValidationUtils.validate_range(50, 0, 100) is True

    def test_validate_range_boundary_lower(self):
        """境界値（下限）"""
        assert ValidationUtils.validate_range(0, 0, 100) is True

    def test_validate_range_boundary_upper(self):
        """境界値（上限）"""
        assert ValidationUtils.validate_range(100, 0, 100) is True

    def test_validate_range_out_of_range_lower(self):
        """範囲外（下限未満）"""
        with patch('app.services.auto_strategy.utils.validation_utils.logger') as mock_logger:
            result = ValidationUtils.validate_range(-10, 0, 100)
            assert result is False
            mock_logger.warning.assert_called_with("値が範囲外です: -10 (範囲: 0-100)")

    def test_validate_range_out_of_range_upper(self):
        """範囲外（上限超過）"""
        with patch('app.services.auto_strategy.utils.validation_utils.logger') as mock_logger:
            result = ValidationUtils.validate_range(150, 0, 100)
            assert result is False
            mock_logger.warning.assert_called_with("値が範囲外です: 150 (範囲: 0-100)")

    def test_validate_range_float_values(self):
        """float値のテスト"""
        assert ValidationUtils.validate_range(50.5, 0.0, 100.0) is True

    def test_validate_required_fields_normal(self):
        """必須フィールドがすべて存在する場合"""
        data = {"field1": "value1", "field2": "value2", "field3": ""}
        is_valid, missing = ValidationUtils.validate_required_fields(data, ["field1", "field2"])
        assert is_valid is True
        assert missing == []

    def test_validate_required_fields_missing_single(self):
        """必須フィールドが1つ不足"""
        data = {"field1": "value1"}
        required_fields = ["field1", "field2"]
        with patch('app.services.auto_strategy.utils.validation_utils.logger') as mock_logger:
            is_valid, missing = ValidationUtils.validate_required_fields(data, required_fields)
            assert is_valid is False
            assert missing == ["field2"]
            mock_logger.warning.assert_called_with("必須フィールドが不足しています: ['field2']")

    def test_validate_required_fields_missing_multiple(self):
        """必須フィールドが複数不足"""
        data = {}
        required_fields = ["field1", "field2", "field3"]
        with patch('app.services.auto_strategy.utils.validation_utils.logger') as mock_logger:
            is_valid, missing = ValidationUtils.validate_required_fields(data, required_fields)
            assert is_valid is False
            assert set(missing) == {"field1", "field2", "field3"}  # setにして順序を無視
            mock_logger.warning.assert_called_with("必須フィールドが不足しています: ['field1', 'field2', 'field3']")

    def test_validate_required_fields_none_values(self):
        """None値は不足とみなす"""
        data = {"field1": "value1", "field2": None, "field3": "value3"}
        required_fields = ["field1", "field2", "field3"]
        with patch('app.services.auto_strategy.utils.validation_utils.logger') as mock_logger:
            is_valid, missing = ValidationUtils.validate_required_fields(data, required_fields)
            assert is_valid is False
            assert missing == ["field2"]
            mock_logger.warning.assert_called_with("必須フィールドが不足しています: ['field2']")

    def test_validate_required_fields_empty_required(self):
        """必須フィールドなし"""
        data = {"field1": "value1"}
        is_valid, missing = ValidationUtils.validate_required_fields(data, [])
        assert is_valid is True
        assert missing == []