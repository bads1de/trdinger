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


# =============================================================================
# 指標リスト取得関連テスト
# =============================================================================


class TestIndicatorUtils:
    """Indicator Utilsのテスト"""

    @pytest.fixture
    def mock_registry(self):
        with patch(
            "app.services.auto_strategy.utils.indicator_utils._load_indicator_registry"
        ) as mock_load:
            registry = MagicMock()
            registry._configs = {
                "RSI": MagicMock(category="momentum", indicator_name="RSI", aliases=[]),
                "SMA": MagicMock(category="trend", indicator_name="SMA", aliases=[]),
                "EMA": MagicMock(category="trend", indicator_name="EMA", aliases=[]),
                "ATR": MagicMock(
                    category="volatility", indicator_name="ATR", aliases=[]
                ),
                "OBV": MagicMock(category="volume", indicator_name="OBV", aliases=[]),
                "Unknown": MagicMock(
                    category="other", indicator_name="Unknown", aliases=[]
                ),
                # Config with no category should be skipped or handled gracefully
                "Broken": None,
            }
            registry.list_indicators.return_value = list(registry._configs.keys())
            mock_load.return_value = registry
            yield registry

    def test_indicators_by_category(self, mock_registry):
        """カテゴリ別の指標取得"""
        trend = indicators_by_category("trend")
        assert "SMA" in trend
        assert "EMA" in trend
        assert "RSI" not in trend

        momentum = indicators_by_category("momentum")
        assert "RSI" in momentum
        assert "SMA" not in momentum

    @patch("app.services.auto_strategy.utils.indicator_utils.get_volume_indicators")
    @patch("app.services.auto_strategy.utils.indicator_utils.get_momentum_indicators")
    @patch("app.services.auto_strategy.utils.indicator_utils.get_trend_indicators")
    @patch("app.services.auto_strategy.utils.indicator_utils.get_volatility_indicators")
    def test_get_all_indicators(
        self, mock_volatility, mock_trend, mock_momentum, mock_volume
    ):
        """全指標取得"""
        mock_volume.return_value = ["OBV"]
        mock_momentum.return_value = ["RSI"]
        mock_trend.return_value = ["SMA"]
        mock_volatility.return_value = ["ATR"]

        all_inds = get_all_indicators()

        assert "OBV" in all_inds
        assert "RSI" in all_inds
        assert "SMA" in all_inds
        assert "ATR" in all_inds

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

    def test_get_valid_indicator_types(self, mock_registry):
        """有効な指標タイプ一覧"""
        valid = get_valid_indicator_types()

        assert "SMA" in valid
        assert "RSI" in valid
        assert "OBV" in valid


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

    def test_process_thresholds(self):
        """閾値処理ロジックのテスト"""
        thresholds = {"rsi_lt": 30, "rsi_gt": 70, "long_gt": 50, "other": 10}
        processed = IndicatorCharacteristics._process_thresholds(thresholds)

        assert processed["rsi_oversold"] == 30
        assert processed["rsi_overbought"] == 70
        assert processed["long_signal_gt"] == 50
        assert processed["other"] == 10

    def test_extract_oscillator_settings_0_100(self):
        """オシレーター(0-100)設定抽出"""
        char = {}
        config = {"scale_type": "oscillator_0_100"}
        thresholds = {}

        settings = IndicatorCharacteristics._extract_oscillator_settings(
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

        settings = IndicatorCharacteristics._extract_oscillator_settings(
            char, config, thresholds
        )

        assert settings["range"] is None
        assert settings["zero_cross"] is True

    def test_apply_condition_based_settings_oversold(self):
        """条件に基づく設定（買われすぎ/売られすぎ）"""
        settings = {}
        # Test input needs to match substring check
        conditions_input = {"long": "rsi_long_lt_30", "short": "rsi_short_gt_70"}

        updated = IndicatorCharacteristics._apply_condition_based_settings(
            settings, conditions_input, {}
        )
        assert updated.get("oversold_based") is True
        assert updated.get("overbought_based") is True

    def test_get_threshold_from_config(self):
        """設定から閾値取得"""
        yaml_config = {}  # Dummy, not used in logic
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

        # Missing profile fallback
        context_missing = {"threshold_profile": "missing"}
        val_missing = IndicatorCharacteristics.get_threshold_from_config(
            yaml_config, config, "long", context_missing
        )
        assert val_missing is None
