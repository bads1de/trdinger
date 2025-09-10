"""
テスト: BaseConfigクラス

BaseConfigクラスの機能をテストします。
TDD準拠で、基本機能からバグ検出のためのエッジケースまでテストします。
"""

import pytest
from unittest.mock import patch, Mock
from dataclasses import dataclass, field
from typing import Any, Dict, List
import json

from backend.app.services.auto_strategy.config.base import BaseConfig


# テスト用サブクラス
@dataclass
class TestConfig(BaseConfig):
    """テスト用BaseConfigサブクラス"""
    enabled: bool = True
    name: str = "test"
    value: int = 10
    optional_field: str = field(default="")
    list_field: List[str] = field(default_factory=list)

    validation_rules = {
        "required_fields": ["name"],
        "ranges": {
            "value": [0, 100],
        },
        "types": {
            "enabled": bool,
            "name": str,
            "value": int,
        },
    }

    def _custom_validation(self) -> List[str]:
        errors = []
        if self.name == "invalid":
            errors.append("カスタム検証失敗")
        return errors


class TestBaseConfig:
    """BaseConfigクラスのテスト"""

    def test_default_values(self):
        """デフォルト値取得のテスト"""
        config = TestConfig()
        defaults = config.get_default_values()

        assert isinstance(defaults, dict)
        assert "enabled" in defaults
        assert "name" in defaults
        assert defaults["enabled"] is True
        assert defaults["name"] == "test"

    def test_get_default_values_from_fields(self):
        """フィールドからデフォルト値取得のテスト"""
        defaults = TestConfig.get_default_values_from_fields()

        assert isinstance(defaults, dict)
        assert "enabled" in defaults
        assert defaults["enabled"] is True

    def test_get_default_values_from_fields_with_factory(self):
        """field(default_factory=...)のテスト"""
        defaults = TestConfig.get_default_values_from_fields()

        assert "list_field" in defaults
        assert defaults["list_field"] == []

    def test_get_default_values_from_fields_error_handling(self):
        """デフォルト値生成エラーのハンドリング"""
        from dataclasses import field, MISSING

        # default_factoryが例外を投げるようにMock作成
        mock_factory = Mock()
        mock_factory.side_effect = Exception("factory error")

        with patch('backend.app.services.auto_strategy.config.base.fields') as mock_fields:
            mock_field = Mock()
            mock_field.name = "error_field"
            mock_field.default = MISSING
            mock_field.default_factory = mock_factory
            mock_fields.return_value = [mock_field]

            with patch('backend.app.services.auto_strategy.config.base.MISSING', MISSING):
                with patch('backend.app.services.auto_strategy.config.base.logger') as mock_logger:
                    config = BaseConfig()
                    defaults = config.get_default_values_from_fields()

                    # 警告ログが呼ばれていることを確認
                    mock_logger.warning.assert_called_once()
                    # エラー時はNoneが返される
                    assert defaults["error_field"] is None

    def test_validate_success(self):
        """正常な検証テスト"""
        config = TestConfig(name="test", value=50)

        is_valid, errors = config.validate()

        assert is_valid is True
        assert errors == []

    def test_validate_required_field_missing(self):
        """必須フィールド欠如テスト"""
        config = TestConfig(name="")  # requiredフィールドを空に

        is_valid, errors = config.validate()

        assert is_valid is False
        assert "必須フィールド 'name' が設定されていません" in errors

    def test_validate_range_out_of_bounds(self):
        """範囲外テスト"""
        config = TestConfig(name="test", value=200)  # 範囲外

        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("value" in error and "範囲" in error for error in errors)

    def test_validate_type_mismatch(self):
        """型ミスマッチテスト"""
        config = TestConfig(name=123)  # intをstrフィールドに

        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("name" in error and "str" in error for error in errors)

    def test_validate_custom_validation_failure(self):
        """カスタム検証失敗テスト"""
        config = TestConfig(name="invalid")

        is_valid, errors = config.validate()

        assert is_valid is False
        assert "カスタム検証失敗" in errors

    def test_validate_exception_handling(self):
        """例外ハンドリングテスト"""
        config = TestConfig()

        with patch.object(config, '_custom_validation', side_effect=Exception("test error")):
            with patch('backend.app.services.auto_strategy.config.base.logger') as mock_logger:
                is_valid, errors = config.validate()

                assert is_valid is False
                assert any("検証処理エラー" in error for error in errors)
                mock_logger.error.assert_called_once()

    def test_from_dict_success(self):
        """正常な辞書からの変換テスト"""
        data = {"enabled": False, "name": "new_name", "value": 75}

        config = TestConfig.from_dict(data)

        assert config.enabled is False
        assert config.name == "new_name"
        assert config.value == 75

    def test_from_dict_invalid_attribute(self):
        """無効な属性を含む辞書からの変換テスト"""
        data = {"invalid_field": "value", "name": "test"}

        config = TestConfig.from_dict(data)

        assert config.name == "test"
        assert not hasattr(config, "invalid_field")

    def test_from_dict_exception_handling(self):
        """辞書変換時の例外ハンドリング"""
        data = {"name": "test"}

        with patch('backend.app.services.auto_strategy.config.base.setattr', side_effect=Exception("setattr error")):
            with patch('backend.app.services.auto_strategy.config.base.logger') as mock_logger:
                result = TestConfig.from_dict(data)

                assert isinstance(result, TestConfig)
                mock_logger.warning.assert_called_once()

    def test_to_dict_success(self):
        """正常な辞書変換テスト"""
        config = TestConfig(name="test", value=42)

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42
        # complex objectの処理テスト
        assert "enabled" in result

    def test_to_dict_exception_handling(self):
        """辞書変換時の例外ハンドリング"""
        config = TestConfig()

        with patch('backend.app.services.auto_strategy.config.base.fields', side_effect=Exception("fields error")):
            with patch('backend.app.services.auto_strategy.config.base.logger') as mock_logger:
                result = config.to_dict()

                assert result == {}
                mock_logger.error.assert_called_once()

    def test_to_json_success(self):
        """JSON変換成功テスト"""
        config = TestConfig(name="test", value=42)

        json_str = config.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["name"] == "test"

    def test_to_json_exception_handling(self):
        """JSON変換時の例外ハンドリング"""
        config = TestConfig()

        with patch.object(config, 'to_dict', side_effect=Exception("to_dict error")):
            with patch('backend.app.services.auto_strategy.config.base.logger') as mock_logger:
                json_str = config.to_json()

                assert json_str == "{}"
                mock_logger.error.assert_called_once()

    def test_from_json_success(self):
        """JSONからの変換成功テスト"""
        json_str = '{"name": "test", "value": 42}'

        config = TestConfig.from_json(json_str)

        assert isinstance(config, TestConfig)
        assert config.name == "test"
        assert config.value == 42

    def test_from_json_invalid_json(self):
        """無効JSONからの変換テスト"""
        invalid_json = '{"name": "test", "value": }'  # 無効JSON

        with patch('backend.app.services.auto_strategy.config.base.logger') as mock_logger:
            result = TestConfig.from_json(invalid_json)

            assert isinstance(result, TestConfig)  # エラー時はデフォルトコンストラクタ
            mock_logger.error.assert_called_once()

    def test_from_json_json_load_failure(self):
        """JSONパース失敗テスト"""
        invalid_json = 'invalid json'

        with patch('backend.app.services.auto_strategy.config.base.logger') as mock_logger:
            result = TestConfig.from_json(invalid_json)

            assert isinstance(result, TestConfig)
            mock_logger.error.assert_called_once()


# pytest.mark.parametrizeを使用したパラメータ化テストも追加可能

@pytest.mark.parametrize("field_name,invalid_value,expected_error", [
    ("value", 150, "範囲"),  # 範囲外
    ("enabled", "string", "bool"),  # 型ミスマッチ
    ("name", 123, "str"),  # 型ミスマッチ
])
def test_validate_parametrized(field_name, invalid_value, expected_error):
    """パラメータ化テスト：異なる無効値での検証"""
    config = TestConfig()
    setattr(config, field_name, invalid_value)

    is_valid, errors = config.validate()

    assert is_valid is False
    assert any(expected_error in error for error in errors)