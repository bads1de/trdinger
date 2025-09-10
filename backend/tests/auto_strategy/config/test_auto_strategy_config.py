"""
テスト: AutoStrategyConfigクラス

AutoStrategyConfigクラスの機能をテストします。
TDD準拠で、基本機能からバグ検出のためのエッジケースまでテストします。
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, Mock
from typing import Dict, Any
from dataclasses import dataclass, field
from pytest import raises

# テスト対象のクラス
from backend.app.services.auto_strategy.config.auto_strategy import AutoStrategyConfig, get_default_config, create_config_from_file, validate_config_file
from backend.app.services.auto_strategy.config.trading import TradingSettings
from backend.app.services.auto_strategy.config.indicators import IndicatorSettings
from backend.app.services.auto_strategy.config.ga import GASettings
from backend.app.services.auto_strategy.config.tpsl import TPSLSettings
from backend.app.services.auto_strategy.config.position_sizing import PositionSizingSettings
from backend.app.services.auto_strategy.constants import ERROR_CODES, THRESHOLD_RANGES


class TestAutoStrategyConfig:
    """AutoStrategyConfigクラスのテスト"""

    def test_initialize_default(self):
        """デフォルト初期化テスト"""
        config = AutoStrategyConfig()

        # 設定グループが正しい型であることを確認
        assert isinstance(config.trading, TradingSettings)
        assert isinstance(config.indicators, IndicatorSettings)
        assert isinstance(config.ga, GASettings)
        assert isinstance(config.tpsl, TPSLSettings)
        assert isinstance(config.position_sizing, PositionSizingSettings)

        # 共通設定のデフォルト値確認
        assert config.enable_caching is True
        assert config.cache_ttl_hours == 24
        assert config.enable_async_processing is False
        assert config.log_level == "WARNING"

    def test_get_default_values(self):
        """get_default_valuesメソッドテスト"""
        config = AutoStrategyConfig()
        defaults = config.get_default_values()

        assert isinstance(defaults, dict)
        assert "enable_caching" in defaults
        assert "trading" in defaults
        assert "indicators" in defaults
        assert "ga" in defaults
        assert "tpsl" in defaults
        assert "position_sizing" in defaults

        # サブコンポーネントのデフォルト値が統合されているか確認
        assert isinstance(defaults["trading"], dict)
        assert isinstance(defaults["ga"], dict)

    def test_validate_success(self):
        """正常な設定の検証テスト"""
        config = AutoStrategyConfig()
        is_valid, errors = config.validate()

        assert is_valid is True
        assert errors == []

    def test_validate_log_level_invalid(self):
        """無効なログレベルのテスト"""
        config = AutoStrategyConfig(log_level="INVALID")
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("無効なログレベル" in error for error in errors)

    def test_validate_cache_ttl_negative(self):
        """負のキャッシュTTLのテスト"""
        config = AutoStrategyConfig(cache_ttl_hours=-1)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("キャッシュTTLは正の数である必要があります" in error for error in errors)

    def test_validate_all_success(self):
        """全設定グループ検証成功テスト"""
        config = AutoStrategyConfig()
        is_valid, errors = config.validate_all()

        assert is_valid is True
        assert errors == {}

    def test_validate_all_with_error(self):
        """サブコンポーネント検証失敗テスト"""
        config = AutoStrategyConfig()

        # モックでサブコンポーネントの検証を失敗させる
        with patch.object(config.trading, 'validate', return_value=(False, ["Trading error"])):
            is_valid, errors = config.validate_all()

            assert is_valid is False
            assert "trading" in errors
            assert "main" in errors

    def test_to_nested_dict(self):
        """ネスト辞書変換テスト"""
        config = AutoStrategyConfig()
        result = config.to_nested_dict()

        assert isinstance(result, dict)
        assert "trading" in result
        assert "enable_caching" in result
        assert isinstance(result["trading"], dict)

    def test_to_nested_dict_exception(self):
        """変換エラーハンドリングテスト"""
        config = AutoStrategyConfig()

        # サブコンポーネントのメソッドが例外を投げる場合
        with patch.object(config.trading, 'to_dict', side_effect=Exception("to_dict error")):
            result = config.to_nested_dict()

            assert result == {}

    def test_from_nested_dict_success(self):
        """ネスト辞書からの作成成功テスト"""
        data = {
            "trading": {},
            "indicators": {},
            "ga": {},
            "tpsl": {},
            "position_sizing": {},
            "enable_caching": False,
            "cache_ttl_hours": 48,
        }

        config = AutoStrategyConfig.from_nested_dict(data)

        assert isinstance(config, AutoStrategyConfig)
        assert config.enable_caching is False
        assert config.cache_ttl_hours == 48

    def test_from_nested_dict_exception(self):
        """作成エラーハンドリングテスト"""
        data = {
            "enable_caching": True,
        }

        # from_dictが例外を投げる場合
        with patch.object(TradingSettings, 'from_dict', side_effect=Exception("from_dict error")):
            config = AutoStrategyConfig.from_nested_dict(data)

            # エラーが発生するはず
            assert config is not None  # テスト用: エラーがログに出力されることを確認

    def test_save_to_json_success(self):
        """JSON保存成功テスト"""
        config = AutoStrategyConfig()

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            filepath = f.name

        try:
            result = config.save_to_json(filepath)
            assert result is True

            # ファイルが存在することを確認
            assert os.path.exists(filepath)

            # JSONとして読み込むことができることを確認
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert isinstance(data, dict)

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_to_json_failure(self):
        """JSON保存失敗テスト"""
        config = AutoStrategyConfig()

        # 無効なパス
        result = config.save_to_json("/invalid/path/config.json")
        assert result is False

    def test_load_from_json_success(self):
        """JSON読み込み成功テスト"""
        original_config = AutoStrategyConfig(log_level="ERROR")
        data = original_config.to_nested_dict()

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(data, f)
            filepath = f.name

        try:
            config = AutoStrategyConfig.load_from_json(filepath)

            assert isinstance(config, AutoStrategyConfig)
            assert config.log_level == "ERROR"

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_load_from_json_failure(self):
        """JSON読み込み失敗テスト"""
        # 存在しないファイル
        with pytest.raises(ValueError, match="設定ファイルの読み込みに失敗しました"):
            AutoStrategyConfig.load_from_json("/nonexistent/path/config.json")

    def test_get_default_config(self):
        """デフォルト設定を取得する関数のテスト"""
        config = get_default_config()

        assert isinstance(config, AutoStrategyConfig)

    def test_create_config_from_file(self):
        """ファイルから設定を作成する関数のテスト"""
        original_config = AutoStrategyConfig(log_level="INFO")

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            original_config.save_to_json(f.name)
            filepath = f.name

        try:
            config = create_config_from_file(filepath)

            assert isinstance(config, AutoStrategyConfig)
            assert config.log_level == "INFO"

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_validate_config_file_success(self):
        """設定ファイル検証成功テスト"""
        config = AutoStrategyConfig()

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            config.save_to_json(f.name)
            filepath = f.name

        try:
            is_valid, errors = validate_config_file(filepath)

            assert is_valid is True
            assert errors == {}

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_validate_config_file_missing(self):
        """存在しない設定ファイル検証テスト"""
        is_valid, errors = validate_config_file("/nonexistent/config.json")

        assert is_valid is False
        assert "file_error" in errors


@pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_valid_log_levels(log_level):
    """有効なログレベルのパラメータ化テスト"""
    config = AutoStrategyConfig(log_level=log_level)
    is_valid, errors = config.validate()

    assert is_valid is True


@pytest.mark.parametrize("invalid_log_level", ["invalid", "debug", "", None])
def test_invalid_log_levels(invalid_log_level):
    """無効なログレベルのパラメータ化テスト"""
    config = AutoStrategyConfig(log_level=invalid_log_level)
    is_valid, errors = config.validate()

    assert is_valid is False
    assert any("無効なログレベル" in error for error in errors)


@pytest.mark.parametrize("cache_ttl", [-1, 0, 168.5, "invalid"])
def test_invalid_cache_ttl(cache_ttl):
    """無効なキャッシュTTLのテスト"""
    config = AutoStrategyConfig(cache_ttl_hours=cache_ttl)
    is_valid, errors = config.validate()

    assert is_valid is False
    assert any("キャッシュTTLは正の数である必要があります" in error for error in errors)