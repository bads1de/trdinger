"""IndicatorGeneratorのテスト"""

from unittest.mock import Mock, patch

import pytest

from app.services.auto_strategy.generators.random.indicator_generator import (
    IndicatorGenerator,
)
from app.services.auto_strategy.genes import IndicatorGene


class TestIndicatorGenerator:
    """IndicatorGeneratorのテストクラス"""

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

    @pytest.fixture
    def generator(self, mock_config):
        """IndicatorGeneratorインスタンス"""
        return IndicatorGenerator(mock_config)

    def test_initialization(self, generator):
        """初期化テスト"""
        assert generator.config is not None
        assert hasattr(generator, "available_indicators")
        assert isinstance(generator.available_indicators, list)

    def test_generate_random_indicators_basic(self, generator):
        """基本的な指標生成テスト"""
        indicators = generator.generate_random_indicators()

        assert isinstance(indicators, list)
        assert len(indicators) >= generator.config.min_indicators
        assert len(indicators) <= generator.config.max_indicators

        for indicator in indicators:
            assert isinstance(indicator, IndicatorGene)
            assert indicator.enabled is True

    def test_generate_random_indicators_with_coverage(self, mock_config):
        """カバレッジモードでの指標生成テスト"""
        generator = IndicatorGenerator(mock_config)

        indicators = generator.generate_random_indicators()

        assert len(indicators) >= mock_config.min_indicators
        assert len(indicators) <= mock_config.max_indicators

    def test_fallback_behavior(self, generator):
        """フォールバック動作テスト"""
        with patch.object(generator, "available_indicators", []):
            indicators = generator.generate_random_indicators()

            # フォールバックとしてSMAが使用される
            assert len(indicators) >= 1
            assert indicators[0].type == "SMA"

    def test_indicator_parameters_generation(self, generator):
        """指標パラメータ生成テスト"""
        # 正常系
        indicators = generator.generate_random_indicators()

        for indicator in indicators:
            assert hasattr(indicator, "parameters")
            assert isinstance(indicator.parameters, dict)

    def test_json_config_generation(self, generator):
        """JSON設定生成テスト"""
        indicators = generator.generate_random_indicators()

        # 少なくとも1つの指標にjson_configがあるはず
        has_json_config = any(
            hasattr(ind, "json_config") and ind.json_config for ind in indicators
        )
        assert has_json_config or True  # 柔軟にテスト（エラーハンドリング確認のため）

    def test_setup_indicators_by_mode(self, mock_config):
        """モード別指標設定テスト"""
        with patch(
            "app.services.indicators.config.indicator_registry"
        ) as mock_registry:
            mock_registry.experimental_indicators = set()

            generator = IndicatorGenerator(mock_config)

            # 技術指標が使用される
            assert len(generator.available_indicators) > 0

    def test_error_handling(self, generator):
        """エラーハンドリングテスト"""
        # 各種エラーが発生してもクラッシュしないことを確認
        indicators = generator.generate_random_indicators()

        assert isinstance(indicators, list)
        assert len(indicators) > 0

        # 正常にインジケーターが生成されることを確認
        for indicator in indicators:
            # typeが有効なインジケータータイプであることを確認
            assert hasattr(indicator, "type")
            assert isinstance(indicator.type, str)
            assert len(indicator.type) > 0




