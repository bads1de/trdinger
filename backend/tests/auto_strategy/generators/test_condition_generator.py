import unittest
from unittest.mock import patch, MagicMock

from app.services.auto_strategy.generators.condition_generator import ConditionGenerator
from app.services.auto_strategy.models import IndicatorGene, Condition, ConditionGroup
from app.services.auto_strategy.constants import StrategyType, IndicatorType
from app.services.auto_strategy.generators.strategies import DifferentIndicatorsStrategy, ComplexConditionsStrategy, IndicatorCharacteristicsStrategy

class TestConditionGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = ConditionGenerator(enable_smart_generation=True)

    def test_init(self):
        """初期化のテスト"""
        self.assertTrue(self.generator.enable_smart_generation)
        self.assertIsNotNone(self.generator.yaml_config)
        self.assertIsNotNone(self.generator.context)

    def test_set_context(self):
        """コンテキスト設定のテスト"""
        self.generator.set_context(timeframe="15m", symbol="BTCUSDT", threshold_profile="conservative")
        self.assertEqual(self.generator.context["timeframe"], "15m")
        self.assertEqual(self.generator.context["symbol"], "BTCUSDT")
        self.assertEqual(self.generator.context["threshold_profile"], "conservative")

    def test_generate_balanced_conditions_with_empty_indicators(self):
        """空の指標リストでの条件生成テスト（フォールバック）"""
        long_conditions, short_conditions, exit_conditions = self.generator.generate_balanced_conditions([])
        self.assertEqual(len(long_conditions), 1)
        self.assertEqual(len(short_conditions), 1)
        self.assertEqual(len(exit_conditions), 0)

    def test_generate_balanced_conditions_with_valid_indicators(self):
        """有効な指標での条件生成テスト"""
        indicators = [IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)]
        long_conditions, short_conditions, exit_conditions = self.generator.generate_balanced_conditions(indicators)
        self.assertIsInstance(long_conditions, list)
        self.assertIsInstance(short_conditions, list)
        self.assertTrue(len(long_conditions) > 0)
        self.assertTrue(len(short_conditions) > 0)

    @patch('app.services.auto_strategy.generators.condition_generator.DifferentIndicatorsStrategy')
    def test_generate_balanced_conditions_with_different_indicators_strategy(self, mock_strategy_cls):
        """DifferentIndicatorsStrategyを使用したテスト"""
        indicators = [IndicatorGene(type="SMA", enabled=True), IndicatorGene(type="RSI", enabled=True)]
        mock_strategy = MagicMock()
        mock_strategy.generate_conditions.return_value = ([Condition("SMA", ">", 25)], [Condition("RSI", "<", 30)], [])
        mock_strategy_cls.return_value = mock_strategy

        with patch.object(self.generator, '_select_strategy_type') as mock_select:
            mock_select.return_value = StrategyType.DIFFERENT_INDICATORS
            long_conditions, short_conditions, exit_conditions = self.generator.generate_balanced_conditions(indicators)
            mock_strategy.generate_conditions.assert_called_once()
            self.assertEqual(len(long_conditions), 1)

    def test_generate_balanced_conditions_with_disabled_indicators(self):
        """無効化された指標でのテスト"""
        indicators = [IndicatorGene(type="SMA", enabled=False)]
        long_conditions, short_conditions, exit_conditions = self.generator.generate_balanced_conditions(indicators)
        self.assertEqual(len(long_conditions), 1)  # フォールバック
        self.assertEqual(len(short_conditions), 1)

    def test__select_strategy_type_mixed_indicators(self):
        """混合指標での戦略タイプ選択"""
        indicators = [
            IndicatorGene(type="SMA", enabled=True),
            IndicatorGene(type="ML_UP_PROB", enabled=True)
        ]
        strategy_type = self.generator._select_strategy_type(indicators)
        self.assertEqual(strategy_type, StrategyType.DIFFERENT_INDICATORS)

    def test__select_strategy_type_single_indicator(self):
        """単一指標での戦略タイプ選択"""
        indicators = [IndicatorGene(type="SMA", enabled=True)]
        strategy_type = self.generator._select_strategy_type(indicators)
        self.assertEqual(strategy_type, StrategyType.COMPLEX_CONDITIONS)

    def test__select_strategy_type_ml_only(self):
        """ML指標のみでの戦略タイプ選択"""
        indicators = [IndicatorGene(type="ML_UP_PROB", enabled=True)]
        strategy_type = self.generator._select_strategy_type(indicators)
        self.assertEqual(strategy_type, StrategyType.INDICATOR_CHARACTERISTICS)

    def test__select_strategy_type_empty_indicators(self):
        """空の指標リストでの戦略タイプ選択"""
        indicators = []
        strategy_type = self.generator._select_strategy_type(indicators)
        self.assertEqual(strategy_type, StrategyType.COMPLEX_CONDITIONS)  # default

    def test__generate_fallback_conditions(self):
        """フォールバック条件生成のテスト"""
        long_cond, short_cond, exit_cond = self.generator._generate_fallback_conditions()
        self.assertEqual(len(long_cond), 1)
        self.assertEqual(len(short_cond), 1)
        self.assertEqual(len(exit_cond), 0)

    def test__generic_long_conditions(self):
        """汎用ロング条件生成のテスト"""
        indicator = IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        conditions = self.generator._generic_long_conditions(indicator)
        self.assertIsInstance(conditions, list)
        if conditions:
            self.assertIsInstance(conditions[0], Condition)

    def test__generic_short_conditions(self):
        """汎用ショート条件生成のテスト"""
        indicator = IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        conditions = self.generator._generic_short_conditions(indicator)
        self.assertIsInstance(conditions, list)

    def test_exceptions_in_generate_balanced_conditions(self):
        """例外発生時のテスト"""
        with patch.object(self.generator, '_select_strategy_type', side_effect=Exception("Test error")):
            with patch.object(self.generator, 'logger') as mock_logger:
                long, short, exits = self.generator.generate_balanced_conditions([IndicatorGene(type="SMA", enabled=True)])
                # フォールバックが呼び出されるはず
                self.assertEqual(len(long), 1)
                self.assertEqual(len(short), 1)
                mock_logger.error.assert_called()

    def test_condition_limiting_in_generate_balanced_conditions(self):
        """条件数の制限テスト"""
        indicators = [IndicatorGene(type="SMA", enabled=True)] * 5  # 複数の指標で条件が多く生成される可能性
        long_conditions, short_conditions, exit_conditions = self.generator.generate_balanced_conditions(indicators)
        # 条件数が2個を超えないはず（random.sampleにより制限）
        self.assertLessEqual(len(long_conditions), 2)
        self.assertLessEqual(len(short_conditions), 2)

    # バグ検出用テスト
    def test_random_choice_empty_list_bug(self):
        """random.choice空リストバグ検出"""
        # TRENDのみの指標なのでMOMENTUMは空、random.choice([])エラー期待
        indicators = [IndicatorGene(type="SMA", enabled=True)]
        with patch.object(self.generator, '_select_strategy_type', return_value=StrategyType.DIFFERENT_INDICATORS):
            with patch.object(self.generator, '_dynamic_classify') as mock_classify:
                mock_classify.return_value = {
                    IndicatorType.MOMENTUM: [],  # Empty list to trigger bug
                    IndicatorType.TREND: indicators,
                    IndicatorType.VOLATILITY: []
                }
                with self.assertRaises(IndexError):  # IndexError from random.choice([])
                    self.generator._generate_different_indicators_strategy(indicators)

    def test_rsi_special_handling(self):
        """RSI特別処理テスト"""
        indicator = IndicatorGene(type="RSI", enabled=True)
        self.generator.context["timeframe"] = "15m"
        conditions = self.generator._create_type_based_conditions(indicator, "long")
        # Threshold should be 30 for 15m long
        if conditions:
            self.assertEqual(conditions[0].right_operand, 30)

    def test_ml_conditions_empty_case(self):
        """ML条件空ケーステスト"""
        indicators = [IndicatorGene(type="UNKNOWN", enabled=True)]
        conditions = self.generator._create_ml_long_conditions(indicators)
        self.assertEqual(len(conditions), 0)

    def test_apply_threshold_context_none_lists(self):
        """apply_threshold_contextのNoneリスト検知"""
        indicators = [IndicatorGene(type="SMA", enabled=True)]
        with patch.object(self.generator, '_get_indicator_type') as mock_get_type:
            mock_get_type.return_value = IndicatorType.MOMENTUM
            with patch.object(self.generator, '_create_momentum_long_conditions') as mock_long:
                with patch.object(self.generator, '_create_momentum_short_conditions') as mock_short:
                    # Mock to return None
                    mock_long.return_value = None
                    mock_short.return_value = None

                    result = self.generator.apply_threshold_context(indicators, {})
                    # Should handle None and provide defaults
                    self.assertIn("long_conditions", result)
                    # Check fallback conditions
                    if result["long_conditions"]:
                        self.assertEqual(len(result["long_conditions"]), 1)
                        self.assertEqual(result["long_conditions"][0].left_operand, "close")

if __name__ == '__main__':
    unittest.main()