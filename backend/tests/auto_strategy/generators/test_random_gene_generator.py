import unittest
from unittest.mock import patch, MagicMock, patch

import random
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.models import IndicatorGene, Condition
from app.services.auto_strategy.models.strategy_models import StrategyGene

class MockConfig:
    """モック設定クラス"""
    def __init__(self):
        self.max_indicators = 3
        self.min_indicators = 1
        self.max_conditions = 5
        self.min_conditions = 1
        self.threshold_ranges = {
            'price_ratio': [0.95, 1.05],
            'oscillator_0_100': [20, 80],
            'volume': [1000, 100000]
        }
        self.price_data_weight = 30
        self.volume_data_weight = 10
        self.oi_fr_data_weight = 5
        self.numeric_threshold_probability = 0.5
        self.min_compatibility_score = 0.5
        self.strict_compatibility_score = 0.8
        self.indicator_mode = "technical_only"
        self.allowed_indicators = ["SMA", "RSI", "MACD"]

class TestRandomGeneGenerator(unittest.TestCase):

    def setUp(self):
        self.config = MockConfig()
        # max_conditions を適正な値に変更（条件生成時の上限制限テスト用）
        self.config.max_conditions = 3
        self.generator = RandomGeneGenerator(self.config)

    def test_init(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.generator.config)
        self.assertIsNotNone(self.generator.available_indicators)
        self.assertIsNotNone(self.generator.smart_condition_generator)
        self.assertTrue(self.generator.enable_smart_generation)

    def test_generate_random_gene(self):
        """ランダム遺伝子生成のテスト"""
        gene = self.generator.generate_random_gene()
        self.assertIsInstance(gene, StrategyGene)
        self.assertIsNotNone(gene.indicators)
        self.assertIsNotNone(gene.long_entry_conditions)
        self.assertIsNotNone(gene.short_entry_conditions)

    def test_generate_random_gene_with_empty_config(self):
        """空の設定でのテスト"""
        mock_config = MockConfig()
        mock_config.allowed_indicators = []
        generator = RandomGeneGenerator(mock_config)
        gene = generator.generate_random_gene()
        # フォールバックが動作するはず
        self.assertIsInstance(gene, StrategyGene)

    def test__generate_random_indicators(self):
        """ランダム指標生成のテスト"""
        indicators = self.generator._generate_random_indicators()
        self.assertIsInstance(indicators, list)
        self.assertGreaterEqual(len(indicators), self.config.min_indicators)
        self.assertLessEqual(len(indicators), self.config.max_indicators)

        # 各指標が有効かチェック
        for ind in indicators:
            self.assertIsInstance(ind, IndicatorGene)
            self.assertIn(ind.type, self.config.allowed_indicators)

    def test__generate_random_conditions(self):
        """ランダム条件生成のテスト"""
        indicators = [IndicatorGene(type="SMA", enabled=True)]
        entry_conditions = self.generator._generate_random_conditions(indicators, "entry")
        exit_conditions = self.generator._generate_random_conditions(indicators, "exit")

        self.assertIsInstance(entry_conditions, list)
        self.assertIsInstance(exit_conditions, list)

        # 最低1つの条件
        self.assertGreaterEqual(len(entry_conditions), 1)
        self.assertGreaterEqual(len(exit_conditions), 1)

        # 各条件がConditionインスタンス
        for cond in entry_conditions + exit_conditions:
            self.assertIsInstance(cond, Condition)

    @patch.object(random, 'choice')
    def test__choose_operand(self, mock_choice):
        """オペランド選択のテスト"""
        indicators = [IndicatorGene(type="SMA", enabled=True)]
        mock_choice.side_effect = ["close", "SMA"]  # 最初の選択でclose、次でSMA
        
        operand = self.generator._choose_operand(indicators)
        self.assertIsNotNone(operand)

    @patch.object(random, 'uniform')
    def test__generate_threshold_value(self, mock_uniform):
        """閾値生成のテスト"""
        mock_uniform.return_value = 0.98
        
        threshold = self.generator._generate_threshold_value("SMA", "long")
        self.assertIsInstance(threshold, float)

    @patch.object(random, 'uniform')
    def test__generate_threshold_value_price_ratio(self, mock_uniform):
        """価格比閾値生成のテスト"""
        mock_uniform.return_value = 0.99
        
        threshold = self.generator._generate_threshold_value("close", "long")
        self.assertIsInstance(threshold, float)
        self.assertGreaterEqual(threshold, 0.95)
        self.assertLessEqual(threshold, 1.05)

    def test__get_safe_threshold(self):
        """安全閾値取得のテスト"""
        threshold = self.generator._get_safe_threshold("price_ratio", [0.95, 1.05])
        self.assertIsInstance(threshold, float)
        self.assertGreaterEqual(threshold, 0.95)
        self.assertLessEqual(threshold, 1.05)

    def test__generate_fallback_condition(self):
        """フォールバック条件生成のテスト"""
        condition_entry = self.generator._generate_fallback_condition("entry")
        condition_exit = self.generator._generate_fallback_condition("exit")

        self.assertIsInstance(condition_entry, Condition)
        self.assertIsInstance(condition_exit, Condition)

    def test_exception_in_generate_random_gene(self):
        """例外発生時のテスト"""
        with patch.object(self.generator, '_generate_random_indicators', side_effect=Exception("Test error")):
            gene = self.generator.generate_random_gene()
            # フォールバックが動作するはず
            self.assertIsInstance(gene, StrategyGene)

    def test_condition_size_limiting(self):
        """条件数制限のテスト"""
        indicators = [IndicatorGene(type="SMA", enabled=True)] * 10  # 多数の指標
        conditions = self.generator._generate_random_conditions(indicators, "entry")
        self.assertLessEqual(len(conditions), self.config.max_conditions)

    def test_setup_indicators_by_mode_technical_only(self):
        """テクニカルオンリー時の指標設定テスト"""
        self.config.indicator_mode = "technical_only"
        generator = RandomGeneGenerator(self.config)
        self.assertIn("SMA", generator.available_indicators)

    def test_setup_indicators_by_mode_ml_only(self):
        """MLオンリー時の指標設定テスト"""
        mock_config = MockConfig()
        mock_config.indicator_mode = "ml_only"
        mock_config.allowed_indicators = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
        generator = RandomGeneGenerator(mock_config)
        self.assertIn("ML_UP_PROB", generator.available_indicators)

    @patch('app.services.auto_strategy.utils.yaml_utils.YamlIndicatorUtils')
    def test_yaml_config_loading(self, mock_yaml):
        """YAML設定読み込みのテスト"""
        mock_yaml.load_yaml_config_for_indicators.return_value = {"SMA": {"thresholds": {"long": 25}}}
        generator = RandomGeneGenerator(self.config)
        # yaml_configが正しく読み込まれるか確認
        self.assertIsNotNone(generator.smart_condition_generator.yaml_config)

    def test_operands_in_condition(self):
        """条件内オペランドのテスト"""
        indicators = [IndicatorGene(type="SMA", enabled=True)]
        from app.services.auto_strategy.constants import DATA_SOURCES
        operators = ["SMA", "close", "RSI"] + DATA_SOURCES
        # 条件生成でオペランドが正しく含まれるか確認
        conditions = self.generator._generate_random_conditions(indicators, "entry")
        for cond in conditions:
            self.assertIn(cond.left_operand, operators + [ind.type for ind in indicators])

    def test_coverage_picking(self):
        """カバー一つ選択のテスト"""
        if hasattr(self.generator, '_coverage_pick'):
            self.assertIsNone(self.generator._coverage_pick)

if __name__ == '__main__':
    unittest.main()