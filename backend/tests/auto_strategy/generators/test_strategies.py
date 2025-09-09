"""
Test for condition generation strategies.
"""
import pytest
from unittest.mock import Mock, MagicMock

from backend.app.services.auto_strategy.generators.condition_generator import ConditionGenerator
from backend.app.services.auto_strategy.constants import StrategyType
from backend.app.services.auto_strategy.models.strategy_models import IndicatorGene


class TestStrategies:
    """Test strategy pattern implementation for condition generation."""

    @pytest.fixture
    def mock_indicators(self):
        """Mock indicators for testing."""
        return [
            Mock(type="RSI", enabled=True),
            Mock(type="SMA", enabled=True),
            Mock(type="EMA", enabled=True),
        ]

    @pytest.fixture
    def condition_generator(self):
        """Create ConditionGenerator with smart generation enabled."""
        generator = ConditionGenerator(enable_smart_generation=True)
        generator.set_context(
            timeframe="4h",
            symbol="BTC/USDT",
            threshold_profile="normal"
        )
        return generator

    def test_select_strategy_type_different_indicators(self, condition_generator, mock_indicators):
        """Test strategy type selection for mixed indicators."""
        # Test with multiple indicator types
        mock_indicators.extend([Mock(type="MACD", enabled=True), Mock(type="STOCH", enabled=True)])

        strategy_type = condition_generator._select_strategy_type(mock_indicators)
        assert strategy_type == StrategyType.DIFFERENT_INDICATORS

    def test_generate_different_indicators_strategy(self, condition_generator, mock_indicators):
        """Test DifferentIndicators strategy generation."""
        longs, shorts, exits = condition_generator._generate_different_indicators_strategy(mock_indicators)
        assert longs or shorts  # Should generate at least some conditions
        assert all(isinstance(cond, dict) or hasattr(cond, 'left_operand') for cond in longs + shorts)

    def test_generate_complex_conditions_strategy(self, condition_generator, mock_indicators):
        """Test ComplexConditions strategy generation."""
        longs, shorts, exits = condition_generator._generate_complex_conditions_strategy(mock_indicators)
        assert isinstance(longs, list)
        assert isinstance(shorts, list)
        assert isinstance(exits, list)

    def test_generate_indicator_characteristics_strategy(self, condition_generator):
        """Test IndicatorCharacteristics strategy generation with ML indicators."""
        ml_indicators = [
            Mock(type="ML_UP_PROB", enabled=True),
            Mock(type="ML_DOWN_PROB", enabled=True),
        ]

        longs, shorts, exits = condition_generator._generate_indicator_characteristics_strategy(ml_indicators)
        assert longs or shorts
        # Should contain ML-specific conditions
        all_conditions = longs + shorts
        assert any("ML_" in str(cond.left_operand if hasattr(cond, 'left_operand') else "") or
                  "ML_" in str(cond) for cond in all_conditions)