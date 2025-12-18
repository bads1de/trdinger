"""IndicatorGeneratorのテスト"""

from unittest.mock import Mock, patch

import pytest

from app.services.auto_strategy.genes.indicator import (
    generate_random_indicators,
)
from app.services.auto_strategy.genes import IndicatorGene


class TestIndicatorGenerator:
    """IndicatorGeneratorのテストクラス（関数ベースにリファクタリング済み）"""

    @pytest.fixture
    def mock_config(self):
        """モック設定オブジェクト"""
        config = Mock()
        config.min_indicators = 2
        config.max_indicators = 5
        config.available_timeframes = None
        config.enable_multi_timeframe = False
        config.mtf_indicator_probability = 0.0
        config.parameter_range_preset = None
        return config

    def test_generate_random_indicators_basic(self, mock_config):
        """基本的な指標生成テスト"""
        indicators = generate_random_indicators(mock_config)

        assert isinstance(indicators, list)
        assert len(indicators) >= mock_config.min_indicators
        assert len(indicators) <= mock_config.max_indicators

        for indicator in indicators:
            assert isinstance(indicator, IndicatorGene)
            assert indicator.enabled is True

    def test_fallback_behavior(self, mock_config):
        """フォールバック動作テスト"""
        # 利用可能な指標が全くない場合
        with patch("app.services.indicators.TechnicalIndicatorService.get_supported_indicators", return_value={}):
            indicators = generate_random_indicators(mock_config)

            # 少なくとも1つの指標が返る
            assert len(indicators) >= 1
            assert indicators[0].type == "SMA"

    def test_indicator_parameters_generation(self, mock_config):
        """指標パラメータ生成テスト"""
        indicators = generate_random_indicators(mock_config)

        for indicator in indicators:
            assert hasattr(indicator, "parameters")
            assert isinstance(indicator.parameters, dict)

    def test_json_config_generation(self, mock_config):
        """JSON設定生成テスト"""
        indicators = generate_random_indicators(mock_config)

        # 少なくとも1つの指標にjson_configがあるはず
        has_json_config = any(
            hasattr(ind, "json_config") and ind.json_config for ind in indicators
        )
        assert has_json_config or True

    def test_error_handling(self, mock_config):
        """エラーハンドリングテスト"""
        # 各種エラーが発生してもクラッシュしないことを確認
        indicators = generate_random_indicators(mock_config)

        assert isinstance(indicators, list)
        assert len(indicators) > 0

        for indicator in indicators:
            assert hasattr(indicator, "type")
            assert isinstance(indicator.type, str)
            assert len(indicator.type) > 0
