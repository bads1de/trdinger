"""
テスト - IndicatorGenerator
"""

import pytest
from unittest.mock import Mock

from backend.app.services.auto_strategy.generators.random.indicator_generator import IndicatorGenerator
from backend.app.services.auto_strategy.models.strategy_models import IndicatorGene


class TestIndicatorGenerator:
    """IndicatorGenerator のテスト"""

    def test_indicator_generator_initialization(self):
        """初期化テスト"""
        config = Mock()
        config.indicator_mode = "technical_only"
        config.allowed_indicators = ["SMA", "EMA"]

        generator = IndicatorGenerator(config)
        assert generator.config == config
        assert len(generator.available_indicators) > 0

    def test_generate_random_indicators(self):
        """指標生成テスト"""
        config = Mock()
        config.indicator_mode = "technical_only"
        config.max_indicators = 3
        config.min_indicators = 1

        generator = IndicatorGenerator(config)
        indicators = generator.generate_random_indicators()

        assert isinstance(indicators, list)
        assert len(indicators) >= config.min_indicators
        assert len(indicators) <= config.max_indicators
        for ind in indicators:
            assert isinstance(ind, IndicatorGene)
            assert ind.enabled is True

    def test_setup_indicators_by_mode(self):
        """指標モード設定テスト"""
        config = Mock()
        config.indicator_mode = "technical_only"
        config.allowed_indicators = ["SMA", "EMA"]

        generator = IndicatorGenerator(config)
        indicators = generator._setup_indicators_by_mode(config)

        assert isinstance(indicators, list)
        assert len(indicators) > 0

    def test_experimental_indicators_filtering(self):
        """実験的指標フィルタリングテスト"""
        config = Mock()
        config.indicator_mode = "technical_only"
        config.allowed_indicators = None  # allowed_indicators が指定されていない場合

        generator = IndicatorGenerator(config)
        available_indicators = generator.available_indicators

        # experimental_indicators がフィルタリングされていることを確認
        from backend.app.services.indicators.config.indicator_config import indicator_registry
        experimental = indicator_registry.experimental_indicators

        # available_indicators に experimental の指標が含まれていないことを確認
        for exp_indicator in experimental:
            assert exp_indicator not in available_indicators, f"Experimental indicator {exp_indicator} should be filtered out"