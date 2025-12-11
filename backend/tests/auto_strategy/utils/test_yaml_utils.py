"""
YAML関連ユーティリティのテスト
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from backend.app.services.auto_strategy.utils.yaml_utils import (
    YamlLoadUtils,
    YamlIndicatorUtils,
)


class TestYamlLoadUtils:
    """YamlLoadUtilsのテスト"""

    def test_load_yaml_config_success(self):
        """正常なYAML読み込み"""
        yaml_content = """
        indicators:
          RSI:
            type: RSI
            scale_type: oscillator_0_100
            thresholds:
              normal:
                long_lt: 30
            conditions:
              long: "RSI < 30"
        """
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("pathlib.Path.exists", return_value=True):
                config = YamlLoadUtils.load_yaml_config("test.yaml")
                assert "indicators" in config
                assert "RSI" in config["indicators"]
                assert config["indicators"]["RSI"]["type"] == "RSI"

    def test_load_yaml_config_file_not_found(self):
        """ファイルが存在しない場合"""
        with patch("pathlib.Path.exists", return_value=False):
            config = YamlLoadUtils.load_yaml_config(
                "not_exist.yaml", fallback={"default": True}
            )
            assert config == {"default": True}

    def test_load_yaml_config_invalid_yaml(self):
        """不正なYAML構文"""
        yaml_content = "indicators: [unclosed list"
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("pathlib.Path.exists", return_value=True):
                config = YamlLoadUtils.load_yaml_config(
                    "invalid.yaml", fallback={"fallback": True}
                )
                assert config == {"fallback": True}

    def test_validate_yaml_config_valid(self):
        """有効な設定の検証"""
        config = {
            "indicators": {
                "RSI": {
                    "type": "RSI",
                    "scale_type": "oscillator",
                    "thresholds": {"normal": {}},
                    "conditions": {"long": "cond"},
                }
            }
        }
        is_valid, errors = YamlLoadUtils.validate_yaml_config(config)
        assert is_valid
        assert len(errors) == 0

    def test_validate_yaml_config_missing_section(self):
        """必須セクション欠如"""
        config = {"other": {}}
        is_valid, errors = YamlLoadUtils.validate_yaml_config(config)
        assert not is_valid
        assert any("indicatorsセクションが必須" in e for e in errors)

    def test_validate_yaml_config_invalid_structure(self):
        """不正な構造"""
        config = {"indicators": {"RSI": "not a dict"}}
        is_valid, errors = YamlLoadUtils.validate_yaml_config(config)
        assert not is_valid
        assert any("辞書形式である必要" in e for e in errors)


class TestYamlIndicatorUtils:
    """YamlIndicatorUtilsのテスト"""

    def test_process_thresholds(self):
        """閾値処理ロジックのテスト"""
        thresholds = {"rsi_lt": 30, "rsi_gt": 70, "long_gt": 50, "other": 10}
        processed = YamlIndicatorUtils._process_thresholds(thresholds)

        assert processed["rsi_oversold"] == 30
        assert processed["rsi_overbought"] == 70
        assert processed["long_signal_gt"] == 50
        assert processed["other"] == 10

    def test_extract_oscillator_settings_0_100(self):
        """オシレーター(0-100)設定抽出"""
        char = {}
        config = {"scale_type": "oscillator_0_100"}
        thresholds = {}

        settings = YamlIndicatorUtils._extract_oscillator_settings(
            char, config, thresholds
        )

        assert settings["range"] == (0, 100)
        assert settings["oversold_threshold"] == 30
        assert settings["overbought_threshold"] == 70

    def test_extract_oscillator_settings_centered(self):
        """オシレーター(中心0)設定抽出"""
        char = {}
        config = {"scale_type": "momentum_zero_centered"}
        thresholds = {}

        settings = YamlIndicatorUtils._extract_oscillator_settings(
            char, config, thresholds
        )

        assert settings["range"] is None
        assert settings["zero_cross"] is True

    def test_apply_condition_based_settings_oversold(self):
        """条件に基づく設定（買われすぎ/売られすぎ）"""
        settings = {}
        conditions = {
            "long": "indicator < 30",  # long_lt containing string
            "short": "indicator > 70",  # short_gt containing string
        }
        # Note: Implementation checks for "long_lt" and "short_gt" substrings in the condition string
        # Let's match the check string exactly if needed, or ensuring substrings are present
        # The logic is: if "long_lt" in str(conditions.get("long", "")) ...

        # Test input needs to match substring check
        conditions_input = {"long": "rsi_long_lt_30", "short": "rsi_short_gt_70"}

        updated = YamlIndicatorUtils._apply_condition_based_settings(
            settings, conditions_input, {}
        )
        assert updated.get("oversold_based") is True
        assert updated.get("overbought_based") is True

    def test_get_threshold_from_yaml(self):
        """YAMLから閾値取得"""
        yaml_config = {}  # Dummy, not used in logic
        config = {"thresholds": {"normal": {"long_gt": 0.5, "short_lt": -0.5}}}
        context = {"threshold_profile": "normal"}

        # Long side
        val = YamlIndicatorUtils.get_threshold_from_yaml(
            yaml_config, config, "long", context
        )
        assert val == 0.5

        # Short side
        val = YamlIndicatorUtils.get_threshold_from_yaml(
            yaml_config, config, "short", context
        )
        assert val == -0.5

        # Missing profile fallback
        context_missing = {"threshold_profile": "missing"}
        val_missing = YamlIndicatorUtils.get_threshold_from_yaml(
            yaml_config, config, "long", context_missing
        )
        assert val_missing is None
