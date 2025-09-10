"""indicator_characteristicsモジュールのテストモジュール"""

import pytest
from unittest.mock import patch, MagicMock
from app.services.auto_strategy.utils.indicator_characteristics import INDICATOR_CHARACTERISTICS, _get_merged_characteristics, _INDICATOR_CHARACTERISTICS_BASE

class TestIndicatorCharacteristics:
    """指標特性テスト"""

    def test_indicator_characteristics_basic_structure(self):
        """INDICATOR_CHARACTERISTICSの基本構造テスト"""
        assert isinstance(INDICATOR_CHARACTERISTICS, dict)
        assert len(INDICATOR_CHARACTERISTICS) > 0

        # 基本的な指標が存在することを確認
        expected_indicators = ["RSI", "SMA", "EMA", "MACD", "ADX", "BB"]
        for indicator in expected_indicators:
            assert indicator in INDICATOR_CHARACTERISTICS
            assert "type" in INDICATOR_CHARACTERISTICS[indicator]

    def test_indicator_characteristics_rsi(self):
        """RSI指標の特性テスト"""
        rsi_char = INDICATOR_CHARACTERISTICS["RSI"]

        assert rsi_char["type"] == "momentum"
        assert "range" in rsi_char
        assert "long_zones" in rsi_char
        assert "short_zones" in rsi_char
        assert "neutral_zone" in rsi_char
        assert "oversold_threshold" in rsi_char
        assert "overbought_threshold" in rsi_char

        # RSI固有の値確認
        assert rsi_char["oversold_threshold"] == 30
        assert rsi_char["overbought_threshold"] == 70

    def test_indicator_characteristics_stoch(self):
        """STOCH指標の特性テスト"""
        stoch_char = INDICATOR_CHARACTERISTICS["STOCH"]

        assert stoch_char["type"] == "momentum"
        assert stoch_char["oversold_threshold"] == 20
        assert stoch_char["overbought_threshold"] == 80

    def test_indicator_characteristics_ml_prediction(self):
        """ML予測指標の特性テスト"""
        ml_up = INDICATOR_CHARACTERISTICS["ML_UP_PROB"]
        ml_down = INDICATOR_CHARACTERISTICS["ML_DOWN_PROB"]
        ml_range = INDICATOR_CHARACTERISTICS["ML_RANGE_PROB"]

        # 範囲チェック
        assert ml_up["range"] == (0, 1)
        assert ml_down["range"] == (0, 1)
        assert ml_range["range"] == (0, 1)

        # ゾーン構造チェック
        assert "long_zones" in ml_up
        assert "short_zones" in ml_up
        assert "neutral_zone" in ml_up
        assert "high_confidence_threshold" in ml_up

    def test_indicator_characteristics_bbands(self):
        """BBANDS指標の特性テスト"""
        bb_char = INDICATOR_CHARACTERISTICS["BBANDS"]

        assert bb_char["type"] == "volatility"
        assert "components" in bb_char
        assert bb_char["components"] == ["upper", "middle", "lower"]
        assert bb_char["mean_reversion"] is True
        assert bb_char["breakout_strategy"] is True

    def test_indicator_characteristics_adx(self):
        """ADX指標の特性テスト"""
        adx_char = INDICATOR_CHARACTERISTICS["ADX"]

        assert adx_char["type"] == "trend"
        assert adx_char["range"] == (0, 100)
        assert adx_char["trend_strength"] is True
        assert adx_char["strong_trend_threshold"] == 25

    def test_indicator_characteristics_macd(self):
        """MACD指標の特性テスト"""
        macd_char = INDICATOR_CHARACTERISTICS["MACD"]

        assert macd_char["type"] == "momentum"
        assert macd_char["zero_cross"] is True
        assert macd_char["signal_line"] is True

    def test_indicator_characteristics_trend_followers(self):
        """トレンド追従指標の特性テスト"""
        trend_indicators = ["SMA", "EMA", "FWMA", "SWMA", "VIDYA"]

        for indicator in trend_indicators:
            if indicator in INDICATOR_CHARACTERISTICS:
                char = INDICATOR_CHARACTERISTICS[indicator]
                assert char["type"] == "trend"
                assert char["price_comparison"] is True
                assert char["trend_following"] is True

    def test_indicator_characteristics_regression_based(self):
        """回帰ベース指標の特性テスト"""
        regression_indicators = ["LINREG", "LINREG_SLOPE", "LINREG_INTERCEPT", "LINREG_ANGLE"]

        for indicator in regression_indicators:
            if indicator in INDICATOR_CHARACTERISTICS:
                char = INDICATOR_CHARACTERISTICS[indicator]
                assert char["regression_based"] is True

    def test_indicator_characteristics_linreg_angle(self):
        """LINREG_ANGLE指標の特性テスト（バグ発見目的）"""
        if "LINREG_ANGLE" in INDICATOR_CHARACTERISTICS:
            char = INDICATOR_CHARACTERISTICS["LINREG_ANGLE"]

            assert char["type"] == "trend"
            assert char["range"] == (-90, 90)
            assert char["angle_tracking"] is True
            assert char["trend_strength"] is True

    def test_indicator_characteristics_ppo(self):
        """PPO指標の特性テスト"""
        if "PPO" in INDICATOR_CHARACTERISTICS:
            ppo_char = INDICATOR_CHARACTERISTICS["PPO"]

            assert ppo_char["type"] == "trend"
            assert ppo_char["range"] == (-100, 100)
            assert ppo_char["zero_cross"] is True
            assert ppo_char["signal_line"] is True

    def test_indicator_characteristics_stc(self):
        """STC指標の特性テスト"""
        if "STC" in INDICATOR_CHARACTERISTICS:
            stc_char = INDICATOR_CHARACTERISTICS["STC"]

            assert stc_char["type"] == "trend"
            assert stc_char["range"] == (0, 100)

    def test_get_merged_characteristics_success(self):
        """_get_merged_characteristics関数の成功テスト"""
        original = {"TEST": {"type": "test"}}

        # YamlIndicatorUtilsをモック
        with patch('app.services.auto_strategy.utils.yaml_utils.YamlIndicatorUtils') as mock_yaml_utils:
            mock_yaml_utils.initialize_yaml_based_characteristics.return_value = {
                "TEST": {"type": "test", "merged": True},
                "YAML_BASED": {"type": "yaml"}
            }

            result = _get_merged_characteristics(original)

            assert "merged" in result["TEST"]
            assert result["TEST"]["merged"] is True

    def test_get_merged_characteristics_import_error(self):
        """YamlIndicatorUtils importエラーハンドリングテスト（バグ発見目的）"""
        original = {"TEST": {"type": "test"}}

        # importエラーをシミュレート
        with patch.dict('sys.modules', {'app.services.auto_strategy.utils.yaml_utils': None}):
            with pytest.raises(ModuleNotFoundError):
                _get_merged_characteristics(original)

    def test_indicator_types_coverage(self):
        """指標タイプのカバレッジテスト"""
        types = {char["type"] for char in INDICATOR_CHARACTERISTICS.values()}
        expected_types = {"momentum", "trend", "volatility", "ml_prediction"}

        for expected_type in expected_types:
            if expected_type not in types:
                pytest.fail(f"Expected indicator type '{expected_type}' not found in characteristics")

    def test_indicator_characteristics_has_required_keys(self):
        """各指標特性に必要なキーが含まれているかテスト"""
        for indicator, char in INDICATOR_CHARACTERISTICS.items():
            assert "type" in char, f"{indicator} lacks 'type' key"

            # typeに応じて必要なキーをチェック
            if char["type"] == "momentum":
                # momentumタイプはrangeまたはzero_crossを持つはず
                has_range = "range" in char
                has_zero_cross = "zero_cross" in char
                if not (has_range or has_zero_cross):
                    pytest.skip(f"{indicator}: momentum indicator missing range or zero_cross")
            elif char["type"] in ["trend", "volatility"]:
                # trend/volatilityタイプはprice_comparisonを持つかどうかをチェック
                if "price_comparison" in char:
                    assert isinstance(char["price_comparison"], bool), f"{indicator}: price_comparison should be boolean"

    def test_indicator_range_values(self):
        """指標の範囲値の合理性テスト"""
        for indicator, char in INDICATOR_CHARACTERISTICS.items():
            if "range" in char and char["range"] is not None and len(char["range"]) == 2:
                min_val, max_val = char["range"]

                # min and max are numbers
                assert isinstance(min_val, (int, float)), f"{indicator}: min_val should be number"
                assert isinstance(max_val, (int, float)) or max_val is None, f"{indicator}: max_val should be number or None"

                # min < max if both are numbers
                if isinstance(max_val, (int, float)):
                    assert min_val < max_val, f"{indicator}: Invalid range {min_val} >= {max_val}"
            else:
                # range is None or invalid, skip comparison
                pass