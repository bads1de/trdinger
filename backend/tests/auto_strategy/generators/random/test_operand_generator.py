"""
ConditionGenerator (Operand Generation features) Tests

Test operand selection logic integrated into ConditionGenerator
"""

import pytest
import random as std_random
from unittest.mock import Mock, patch
from app.services.auto_strategy.generators.condition_generator import ConditionGenerator

class TestConditionGeneratorOperandFeatures:
    """ConditionGeneratorに統合されたオペランド生成機能のテスト"""

    @pytest.fixture
    def generator(self):
        """テスト用ジェネレータ"""
        config = Mock()
        config.price_data_weight = 2
        config.volume_data_weight = 1
        config.oi_fr_data_weight = 1
        config.numeric_threshold_probability = 0.5
        config.min_compatibility_score = 0.1
        config.strict_compatibility_score = 0.8
        config.threshold_ranges = {}

        with patch("app.services.auto_strategy.generators.condition_generator.get_all_indicators", return_value={"RSI", "SMA"}):
            return ConditionGenerator(enable_smart_generation=False, ga_config=config)

    def test_choose_operand_includes_all_sources(self, generator):
        """全てのデータソースが選択肢に含まれるか確認"""
        indicator = Mock()
        indicator.type = "RSI"
        
        # 複数回試行して、各タイプが出現するかチェック
        choices = set()
        for _ in range(100):
            res = generator.choose_operand([indicator])
            choices.add(res)
            
        assert "RSI" in choices
        assert "close" in choices
        assert "volume" in choices
        assert "FundingRate" in choices

    def test_choose_right_operand_logic(self, generator):
        """右オペランド選択ロジック（数値 vs 互換オペランド）のテスト"""
        # 数値が選ばれるケース (probability = 1.0)
        generator.ga_config_obj.numeric_threshold_probability = 1.0
        with patch.object(generator, "generate_threshold_value", return_value=50.0):
            res = generator.choose_right_operand("RSI", [], "entry")
            assert res == 50.0

        # 互換オペランドが選ばれるケース (probability = 0.0)
        generator.ga_config_obj.numeric_threshold_probability = 0.0
        with patch.object(generator, "choose_compatible_operand", return_value="SMA"):
            with patch("app.services.auto_strategy.core.operand_grouping.operand_grouping_system.get_compatibility_score", return_value=0.9):
                res = generator.choose_right_operand("RSI", [], "entry")
                assert res == "SMA"

    def test_generate_threshold_value_scales(self, generator):
        """スケールタイプに応じた閾値生成のテスト"""
        # Oscillator 0-100 (RSI等)
        with patch("app.services.indicators.config.indicator_registry.get_indicator_config") as mock_reg:
            mock_cfg = Mock()
            from app.services.indicators.config.indicator_config import IndicatorScaleType
            mock_cfg.scale_type = IndicatorScaleType.OSCILLATOR_0_100
            mock_reg.return_value = mock_cfg
            
            with patch.object(generator, "_get_safe_threshold", return_value=70.0):
                res = generator.generate_threshold_value("RSI", "entry")
                assert res == 70.0

    def test_get_safe_threshold_config_override(self, generator):
        """設定による閾値範囲の上書きテスト"""
        generator.ga_config_obj.threshold_ranges = {"test_key": [100, 200]}
        
        with patch("random.uniform", return_value=150.0):
            res = generator._get_safe_threshold("test_key", [0, 10])
            assert res == 150.0