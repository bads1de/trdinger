"""
Indicator Utils Tests

指標関連ユーティリティとYAML設定ユーティリティの統合テスト
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open

from app.services.auto_strategy.utils.indicator_utils import (
    indicators_by_category,
    get_all_indicators,
    get_all_indicator_ids,
    get_valid_indicator_types,
    ConfigFileUtils,
    IndicatorCharacteristics,
)
from app.services.indicators.config.indicator_config import indicator_registry


# =============================================================================
# 指標リスト取得関連テスト
# =============================================================================


class TestIndicatorUtils:
    """Indicator Utilsのテスト"""

    def test_indicators_by_category(self):
        """カテゴリ別の指標取得"""
        # 実データに基づいたテスト（多くの指標がcustomに分類されている現状に合わせる）
        custom = indicators_by_category("custom")
        assert len(custom) > 0
        assert any(x in custom for x in ["SMA", "RSI"])

    def test_get_all_indicators(self):
        """全指標取得"""
        all_inds = get_all_indicators()
        assert "RSI" in all_inds
        assert "SMA" in all_inds

    @patch("app.services.auto_strategy.utils.indicator_utils.TechnicalIndicatorService")
    def test_get_all_indicator_ids(self, MockService):
        """指標IDマッピング取得"""
        service = MockService.return_value
        service.get_supported_indicators.return_value = {"RSI": {}, "SMA": {}}

        ids = get_all_indicator_ids()

        assert ids[""] == 0
        assert "RSI" in ids
        assert "SMA" in ids
        assert ids["RSI"] > 0

    def test_get_valid_indicator_types(self):
        """有効な指標タイプ一覧"""
        valid = get_valid_indicator_types()
        assert "SMA" in valid
        assert "RSI" in valid


# =============================================================================
# 設定ファイル操作関連ユーティリティテスト（旧YamlLoadUtils）
# =============================================================================


class TestConfigFileUtils:
    """ConfigFileUtilsのテスト"""

    def test_load_config_success(self):
        """正常な設定読み込み"""
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
                config = ConfigFileUtils.load_config("test.yaml")
                assert "indicators" in config
                assert "RSI" in config["indicators"]
                assert config["indicators"]["RSI"]["type"] == "RSI"

    def test_load_config_file_not_found(self):
        """ファイルが存在しない場合"""
        with patch("pathlib.Path.exists", return_value=False):
            config = ConfigFileUtils.load_config(
                "not_exist.yaml", fallback={"default": True}
            )
            assert config == {"default": True}

    def test_load_config_invalid_yaml(self):
        """不正なYAML構文"""
        yaml_content = "indicators: [unclosed list"
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("pathlib.Path.exists", return_value=True):
                config = ConfigFileUtils.load_config(
                    "invalid.yaml", fallback={"fallback": True}
                )
                assert config == {"fallback": True}

    def test_validate_config_valid(self):
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
        is_valid, errors = ConfigFileUtils.validate_config(config)
        assert is_valid
        assert len(errors) == 0

    def test_validate_config_missing_section(self):
        """必須セクション欠如"""
        config = {"other": {}}
        is_valid, errors = ConfigFileUtils.validate_config(config)
        assert not is_valid
        assert any("indicatorsセクションが必須" in e for e in errors)

    def test_validate_config_invalid_structure(self):
        """不正な構造"""
        config = {"indicators": {"RSI": "not a dict"}}
        is_valid, errors = ConfigFileUtils.validate_config(config)
        assert not is_valid
        assert any("辞書形式である必要" in e for e in errors)


class TestIndicatorCharacteristics:
    """IndicatorCharacteristicsのテスト"""

    def test_load_indicator_config_integration(self):
        """registryからの設定読み込み連携テスト"""
        config = IndicatorCharacteristics.load_indicator_config()
        
        assert "indicators" in config
        assert "RSI" in config["indicators"]
        
        # 必須属性の存在確認
        rsi_config = config["indicators"]["RSI"]
        assert "type" in rsi_config
        assert "scale_type" in rsi_config
        assert "thresholds" in rsi_config

    def test_get_threshold_from_config(self):
        """設定から閾値取得"""
        yaml_config = {}  # Dummy
        config = {"thresholds": {"normal": {"long_gt": 0.5, "short_lt": -0.5}}}
        context = {"threshold_profile": "normal"}

        # Long side
        val = IndicatorCharacteristics.get_threshold_from_config(
            yaml_config, config, "long", context
        )
        assert val == 0.5

        # Short side
        val = IndicatorCharacteristics.get_threshold_from_config(
            yaml_config, config, "short", context
        )
        assert val == -0.5

    def test_get_characteristics_backward_compatibility(self):
        """旧メソッド名の互換性テスト"""
        chars = IndicatorCharacteristics.get_characteristics()
        assert "RSI" in chars
        assert "type" in chars["RSI"]
        assert "scale_type" in chars["RSI"]