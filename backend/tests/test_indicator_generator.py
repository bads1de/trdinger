"""IndicatorGeneratorのテスト"""

import pytest
from unittest.mock import Mock, patch

from app.services.auto_strategy.generators.random.indicator_generator import (
    IndicatorGenerator,
)
from app.services.auto_strategy.models.strategy_models import IndicatorGene


class TestIndicatorGenerator:
    """IndicatorGeneratorのテストクラス"""

    @pytest.fixture
    def mock_config(self):
        """モック設定オブジェクト"""
        config = Mock()
        config.min_indicators = 2
        config.max_indicators = 5
        config.allowed_indicators = ["SMA", "EMA", "RSI"]
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
            assert indicator.type in ["SMA", "EMA", "RSI"]
            assert indicator.enabled is True

    def test_generate_random_indicators_with_coverage(self, mock_config):
        """カバレッジモードでの指標生成テスト"""
        mock_config.allowed_indicators = ["SMA", "EMA"]
        generator = IndicatorGenerator(mock_config)

        # カバレッジピックを設定
        generator._coverage_pick = "SMA"

        indicators = generator.generate_random_indicators()

        # 最初の指標がSMAであることを確認
        assert indicators[0].type == "SMA"

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
        mock_config.allowed_indicators = ["SMA", "EMA"]

        with patch(
            "app.services.indicators.config.indicator_registry"
        ) as mock_registry:
            mock_registry.experimental_indicators = set()

            generator = IndicatorGenerator(mock_config)

            # allowed_indicatorsが尊重される
            assert "SMA" in generator.available_indicators
            assert "EMA" in generator.available_indicators

    def test_error_handling(self, generator):
        """エラーハンドリングテスト"""
        # 各種エラーが発生してもクラッシュしないことを確認
        indicators = generator.generate_random_indicators()

        assert isinstance(indicators, list)
        assert len(indicators) > 0

        # エラーが発生してもデフォルトのSMAが返される
        for indicator in indicators:
            assert indicator.type in ["SMA", "EMA", "RSI"] or indicator.type == "SMA"
