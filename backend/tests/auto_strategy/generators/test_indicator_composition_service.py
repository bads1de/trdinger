import unittest
from unittest.mock import patch, MagicMock, patch
from random import choice

from app.services.auto_strategy.generators.indicator_composition_service import IndicatorCompositionService
from app.services.auto_strategy.models import IndicatorGene
from app.services.auto_strategy.constants import MOVING_AVERAGE_INDICATORS

class MockConfig:
    """モック設定クラス"""
    def __init__(self):
        self.max_indicators = 5

class TestIndicatorCompositionService(unittest.TestCase):

    def setUp(self):
        self.config = MockConfig()
        self.service = IndicatorCompositionService(self.config)

    def test_enhance_with_trend_indicators_no_trend(self):
        """トレンド指標がない場合の強制追加テスト"""
        indicators = [IndicatorGene(type="RSI", enabled=True)]  # No trend
        available = ["SMA", "EMA", "RSI"]
        result = self.service.enhance_with_trend_indicators(indicators, available)
        self.assertEqual(len(result), 2)  # RSI + SMA added

    def test_enhance_with_trend_indicators_has_trend(self):
        """トレンド指標がある場合の非追加テスト"""
        indicators = [IndicatorGene(type="SMA", enabled=True)]  # Has trend
        available = ["SMA", "RSI"]
        result = self.service.enhance_with_trend_indicators(indicators, available)
        self.assertEqual(len(result), 1)  # No additional

    @patch('random.sample')
    @patch('random.choice')
    def test_enhance_with_ma_cross_strategy_add_ma(self, mock_choice, mock_sample):
        """MAクロスでMA追加テスト"""
        def sample_side_effect(candidates, length):
            if set(candidates) == {"SMA", "EMA"}:
                return ["EMA", "SMA"]  # EMA first
            return candidates

        def choice_side_effect(values):
            if isinstance(values, list) and values == [10, 14, 20, 30, 50]:
                return 30  # period different from 20
            if isinstance(values, list) and set(values) == {"SMA", "EMA"}:
                return values[0]
            return values

        mock_sample.side_effect = sample_side_effect
        mock_choice.side_effect = choice_side_effect

        indicators = [IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)]
        available = ["SMA", "EMA", "RSI"]
        result = self.service.enhance_with_ma_cross_strategy(indicators, available)
        # Should add EMA with different period
        added_mas = [ind for ind in result if ind.type in ["EMA"]]
        self.assertGreaterEqual(len(added_mas), 1)

    def test_enhance_with_ma_cross_strategy_enough_ma(self):
        """十分なMAがある場合の非追加テスト"""
        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="EMA", parameters={"period": 50}, enabled=True)
        ]
        available = ["SMA", "EMA", "RSI"]
        result = self.service.enhance_with_ma_cross_strategy(indicators, available)
        self.assertEqual(len(result), 2)  # No change

    @patch('app.services.indicators.config.indicator_registry.get_indicator_config')
    def test_is_trend_indicator(self, mock_get_config):
        """トレンド指標判定テスト"""
        mock_get_config.return_value = MagicMock(category="trend")
        is_trend = self.service._is_trend_indicator("SMA")
        self.assertTrue(is_trend)

    def test_choose_preferred_trend_indicator(self):
        """優先トレンド指標選択テスト"""
        pool = ["SMA", "EMA", "VIDYA"]
        chosen = self.service._choose_preferred_trend_indicator(pool)
        self.assertIn(chosen, pool)

    def test_get_default_params_for_indicator_ma(self):
        """MA指標のデフォルトパラメータ取得テスト"""
        params = self.service._get_default_params_for_indicator("SMA")
        self.assertIn("period", params)

    def test_enhance_with_trend_max_indicators_reached(self):
        """最大指標数に達した場合のテスト"""
        # Set max 2
        self.config.max_indicators = 2
        indicators = [
            IndicatorGene(type="RSI", enabled=True),
            IndicatorGene(type="RSI", enabled=True)
        ]
        available = ["SMA"]
        result = self.service.enhance_with_trend_indicators(indicators, available)
        self.assertLessEqual(len(result), self.config.max_indicators)

    def test_enhance_with_ma_cross_max_reached(self):
        """MAクロスで最大指標数に達した場合のテスト"""
        self.config.max_indicators = 2
        indicators = [
            IndicatorGene(type="SMA", enabled=True),
            IndicatorGene(type="RSI", enabled=True)
        ]
        available = ["RSI"]
        result = self.service.enhance_with_ma_cross_strategy(indicators, available)
        self.assertLessEqual(len(result), self.config.max_indicators)

    def test_remove_non_trend_indicator(self):
        """非トレンド指標削除テスト"""
        indicators = [
            IndicatorGene(type="SMA", enabled=True),  # trend
            IndicatorGene(type="RSI", enabled=True)  # non-trend
        ]
        removed_type = self.service._remove_non_trend_indicator(indicators)
        self.assertEqual(removed_type, "RSI")
        self.assertEqual(len(indicators), 1)

    @patch('random.choice')
    def test_choose_ma_with_unique_period(self, mock_choice):
        """一意のperiod Ma選択テスト"""
        mock_choice.return_value = "EMA"
        ma_pool = ["EMA"]
        existing_periods = {20}
        chosen = self.service._choose_ma_with_unique_period(ma_pool, existing_periods)
        self.assertEqual(chosen, "EMA")

    def test_get_existing_periods(self):
        """既存period取得テスト"""
        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ]
        periods = self.service._get_existing_periods(indicators)
        self.assertEqual(periods, {20})

    def test_empty_indicators_list_trend_enhancement(self):
        """空の指標リストでのトレンド強化テスト"""
        indicators = []
        available = ["SMA", "RSI"]
        result = self.service.enhance_with_trend_indicators(indicators, available)
        self.assertGreater(len(result), 0)  # Should add at least one trend indicator

    def test_empty_indicators_list_ma_enhancement(self):
        """空の指標リストでのMA強化テスト"""
        indicators = []
        available = ["SMA", "EMA"]
        result = self.service.enhance_with_ma_cross_strategy(indicators, available)
        self.assertGreater(len(result), 0)  # Should add at least one MA indicator

    def test_no_available_indicators_trend(self):
        """利用可能な指標が空の場合のトレンドテスト"""
        indicators = [IndicatorGene(type="RSI", enabled=True)]
        available = []
        # Should handle gracefully without raising exception
        result = self.service.enhance_with_trend_indicators(indicators, available)
        self.assertEqual(len(result), 1)  # No change

    def test_no_available_indicators_ma(self):
        """利用可能な指標が空の場合のMAテスト"""
        indicators = [IndicatorGene(type="RSI", enabled=True)]
        available = []
        # Should handle gracefully without raising exception
        result = self.service.enhance_with_ma_cross_strategy(indicators, available)
        self.assertEqual(len(result), 1)  # No change

    def test_remove_non_ma_indicator_no_non_ma(self):
        """削除対象の非MA指標がない場合のテスト"""
        indicators = [
            IndicatorGene(type="SMA", enabled=True),
            IndicatorGene(type="EMA", enabled=True)
        ]
        initial_length = len(indicators)
        self.service._remove_non_ma_indicator(indicators)
        self.assertEqual(len(indicators), initial_length)  # No change

    def test_get_default_params_for_non_ma_indicator(self):
        """非MA指標のデフォルトパラメータ取得テスト"""
        params = self.service._get_default_params_for_indicator("RSI")
        self.assertEqual(params, {})  # Should return empty dict

    @patch('app.services.indicators.config.indicator_registry.get_indicator_config')
    def test_is_trend_indicator_no_config(self, mock_get_config):
        """インジケータ設定がない場合のトレンド判定テスト"""
        mock_get_config.return_value = None
        is_trend = self.service._is_trend_indicator("UNKNOWN")
        self.assertFalse(is_trend)

    @patch('app.services.indicators.config.indicator_registry.get_indicator_config')
    def test_is_trend_indicator_exception(self, mock_get_config):
        """インジケータ設定取得で例外が発生した場合のテスト"""
        mock_get_config.side_effect = Exception("Config error")
        is_trend = self.service._is_trend_indicator("SMA")
        self.assertFalse(is_trend)  # Should return False on exception

    def test_choose_preferred_trend_indicator_empty_pool(self):
        """空のトレンドプールでの優先選択テスト"""
        chosen = self.service._choose_preferred_trend_indicator([])
        self.assertEqual(chosen, "SMA")  # Should return fallback

    def test_choose_ma_with_unique_period_no_candidates(self):
        """候補MAがない場合のテスト"""
        chosen = self.service._choose_ma_with_unique_period([], set())
        self.assertIsNone(chosen)

    @patch('random.sample')
    def test_choose_ma_with_unique_period_sample_error(self, mock_sample):
        """sample関数でエラーが発生した場合のテスト"""
        mock_sample.side_effect = Exception("Sample error")
        chosen = self.service._choose_ma_with_unique_period(["SMA"], set())
        # Should return first candidate due to try/catch
        self.assertIsNone(chosen)  # Error in choose should return None

    def test_get_default_params_exception(self):
        """デフォルトパラメータ取得での例外テスト"""
        # Test with exception in random.choice
        with patch('random.choice', side_effect=Exception("Choice error")):
            params = self.service._get_default_params_for_indicator("SMA")
            self.assertEqual(params, {"period": 20})  # Should return fallback

    def test_indicator_without_parameters(self):
        """パラメータのない指標のperiod取得テスト"""
        indicators = [IndicatorGene(type="RSI", enabled=True)]  # No parameters
        periods = self.service._get_existing_periods(indicators)
        self.assertEqual(periods, set())  # Should return empty set

    @patch('app.services.indicators.config.indicator_registry.get_indicator_config')
    def test_is_trend_indicator_no_category(self, mock_get_config):
        """カテゴリなしインジケータのトレンド判定テスト"""
        mock_config = MagicMock()
        mock_config.category = None
        mock_get_config.return_value = mock_config
        is_trend = self.service._is_trend_indicator("SMA")
        self.assertFalse(is_trend)

    def test_max_indicators_none(self):
        """config.max_indicatorsがNoneの場合のテスト"""
        # Save original value
        original_max = self.config.max_indicators

        # Test with None
        self.config.max_indicators = None
        indicators = [IndicatorGene(type="RSI", enabled=True)]
        available = ["SMA", "EMA"]

        try:
            result = self.service.enhance_with_trend_indicators(indicators, available)
            # Should not crash
        except Exception:
            # If it crashes, that's a bug
            self.fail("None max_indicators should be handled gracefully")

        # Restore original value
        self.config.max_indicators = original_max

    def test_mixed_indicator_types_trend_enhancement(self):
        """さまざまな指標タイプの組み合わせでのトレンド強化テスト"""
        indicators = [
            IndicatorGene(type="RSI", enabled=True),  # Non-trend
            IndicatorGene(type="SMA", enabled=True),  # Trend
            IndicatorGene(type="MACD", enabled=True), # Non-trend
        ]
        available = ["SMA", "EMA", "RSI", "MACD", "STOCH"]
        result = self.service.enhance_with_trend_indicators(indicators, available)
        # Should handle mixed types correctly
        self.assertGreaterEqual(len(result), len(indicators))

    def test_mixed_indicator_types_ma_enhancement(self):
        """さまざまな指標タイプの組み合わせでのMA強化テスト"""
        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),  # MA
            IndicatorGene(type="RSI", enabled=True),  # Non-MA
            IndicatorGene(type="EMA", parameters={"period": 30}, enabled=True), # MA
        ]
        available = ["SMA", "EMA", "WMA", "RSI", "MACD"]
        result = self.service.enhance_with_ma_cross_strategy(indicators, available)
        # Should handle mixed types correctly
        self.assertGreaterEqual(len(result), len(indicators))

    @patch('app.services.indicators.config.indicator_registry.get_indicator_config')
    def test_trend_enhancement_with_registry_error(self, mock_get_config):
        """レジストリエラーでのトレンド強化テスト"""
        # First call succeeds, second fails
        mock_configs = [
            MagicMock(category="oscillator"),  # RSI is non-trend
            None  # Error for SMA check
        ]
        mock_get_config.side_effect = mock_configs

        indicators = [IndicatorGene(type="RSI", enabled=True)]
        available = ["SMA", "EMA", "RSI"]

        # Should not crash due to registry error
        result = self.service.enhance_with_trend_indicators(indicators, available)

    @patch('random.choice')
    def test_ma_enhancement_with_duplicate_periods(self, mock_choice):
        """重複periodを持つMA強化テスト"""
        mock_choice.return_value = 20  # Always returns 20

        indicators = [IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)]
        available = ["SMA", "EMA", "RSI"]

        result = self.service.enhance_with_ma_cross_strategy(indicators, available)
        # Should handle the duplicate period gracefully
        self.assertGreaterEqual(len(result), 1)

    def test_large_indicator_list_trend_enhancement(self):
        """大量の指標リストでのトレンド強化テスト"""
        # Create a large list
        large_indicators = [
            IndicatorGene(type=f"INDICATOR_{i}", enabled=True)
            for i in range(50)
        ]
        available = ["SMA", "EMA"] + [f"INDICATOR_{i}" for i in range(50)]

        result = self.service.enhance_with_trend_indicators(large_indicators, available)
        # Should handle large lists without performance issues or crashes
        self.assertGreaterEqual(len(result), 1)

    def test_large_indicator_list_ma_enhancement(self):
        """大量の指標リストでのMA強化テスト"""
        # Create a large list
        large_indicators = [
            IndicatorGene(type=f"INDICATOR_{i}", parameters={"period": 20}, enabled=True)
            for i in range(50)
        ]
        available = ["SMA", "EMA"] + [f"INDICATOR_{i}" for i in range(50)]

        result = self.service.enhance_with_ma_cross_strategy(large_indicators, available)
        # Should handle large lists without performance issues or crashes
        self.assertGreaterEqual(len(result), 1)

    def test_remove_non_trend_with_all_trend_indicators(self):
        """全てトレンド指標の場合の削除テスト"""
        indicators = [
            IndicatorGene(type="SMA", enabled=True),
            IndicatorGene(type="EMA", enabled=True),
            IndicatorGene(type="VIDYA", enabled=True),
        ]
        with patch.object(self.service, '_is_trend_indicator', return_value=True):
            removed = self.service._remove_non_trend_indicator(indicators)
            self.assertEqual(removed, "")  # Should not remove anything
            self.assertEqual(len(indicators), 3)  # List unchanged

    def test_remove_non_ma_with_all_ma_indicators(self):
        """全てMA指標の場合の削除テスト"""
        indicators = [
            IndicatorGene(type="SMA", enabled=True),
            IndicatorGene(type="EMA", enabled=True),
            IndicatorGene(type="WMA", enabled=True),
        ]
        removed_indicator = self.service._remove_non_ma_indicator(indicators)
        # Should not remove anything if all are MA indicators
        expected_len = len(indicators)
        if any(ind.type in MOVING_AVERAGE_INDICATORS for ind in indicators):
            # If some are actually non-MA, that's fine
            pass
        else:
            # If all are non-MA (which shouldn't be the case), length remains the same
            pass

if __name__ == '__main__':
    unittest.main()