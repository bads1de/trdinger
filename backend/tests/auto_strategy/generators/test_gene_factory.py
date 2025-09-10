import unittest
from unittest.mock import patch, MagicMock
from typing import Any

from app.services.auto_strategy.generators.gene_factory import (
    BaseGeneGenerator,
    SmartGeneGenerator,
    DefaultGeneGenerator,
    GeneGeneratorFactory,
    GeneratorType,
)
from app.services.auto_strategy.models.strategy_models import (
    StrategyGene,
    IndicatorGene,
    Condition,
    TPSLGene,
    PositionSizingGene,
    create_random_tpsl_gene,
    create_random_position_sizing_gene,
)


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


class MockRandomGeneGenerator:
    """モックRandomGeneGeneratorクラス"""
    def __init__(self, config):
        self.config = config

    def _generate_random_indicators(self):
        return [IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)]


class TestGeneGeneratorFactory(unittest.TestCase):

    def setUp(self):
        self.config = MockConfig()

    def test_create_generator_random(self):
        """RANDOMタイプの生成器作成テスト"""
        generator = GeneGeneratorFactory.create_generator(
            GeneratorType.RANDOM, self.config
        )
        self.assertIsNotNone(generator)
        # RandomGeneGeneratorクラスであることを確認
        from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        self.assertIsInstance(generator, RandomGeneGenerator)

    def test_create_generator_smart_without_di(self):
        """SMARTタイプの生成器作成テスト（DIなし）"""
        generator = GeneGeneratorFactory.create_generator(
            GeneratorType.SMART, self.config
        )
        self.assertIsNotNone(generator)
        self.assertIsInstance(generator, SmartGeneGenerator)

    def test_create_generator_smart_with_di(self):
        """SMARTタイプの生成器作成テスト（DIあり）"""
        mock_random_gen = MockRandomGeneGenerator(self.config)
        generator = GeneGeneratorFactory.create_generator(
            GeneratorType.SMART, self.config, mock_random_gen
        )
        self.assertIsNotNone(generator)
        self.assertIsInstance(generator, SmartGeneGenerator)

    def test_create_generator_default(self):
        """DEFAULTタイプの生成器作成テスト"""
        generator = GeneGeneratorFactory.create_generator(
            GeneratorType.DEFAULT, self.config
        )
        self.assertIsNotNone(generator)
        self.assertIsInstance(generator, DefaultGeneGenerator)

    def test_create_generator_unknown_type(self):
        """未知のタイプでのデフォルト作成テスト"""
        with patch('app.services.auto_strategy.generators.gene_factory.logger.warning') as mock_warning:
            generator = GeneGeneratorFactory.create_generator(
                "UnknownType", self.config
            )
            self.assertIsNotNone(generator)
            self.assertIsInstance(generator, DefaultGeneGenerator)
            mock_warning.assert_called_once()


class TestSmartGeneGenerator(unittest.TestCase):

    def setUp(self):
        self.config = MockConfig()

    def test_init_without_di(self):
        """DIなし初期化テスト"""
        generator = SmartGeneGenerator(self.config)
        self.assertEqual(generator.config, self.config)
        self.assertIsNone(generator._random_generator)

    def test_init_with_di(self):
        """DIあり初期化テスト"""
        mock_random_gen = MockRandomGeneGenerator(self.config)
        generator = SmartGeneGenerator(self.config, mock_random_gen)
        self.assertEqual(generator.config, self.config)
        self.assertEqual(generator._random_generator, mock_random_gen)

    def test_generate_indicators_di(self):
        """DIを使用した指標生成テスト"""
        mock_random_gen = MockRandomGeneGenerator(self.config)
        generator = SmartGeneGenerator(self.config, mock_random_gen)

        indicators = generator.generate_indicators()

        self.assertIsInstance(indicators, list)
        self.assertGreater(len(indicators), 0)
        for ind in indicators:
            self.assertIsInstance(ind, IndicatorGene)

    def test_generate_indicators_lazy_init(self):
        """遅延初期化を使用した指標生成テスト"""
        generator = SmartGeneGenerator(self.config)

        with patch('app.services.auto_strategy.generators.random_gene_generator.RandomGeneGenerator') as mock_random_class:
            mock_instance = MagicMock()
            mock_instance._generate_random_indicators.return_value = [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
            ]
            mock_random_class.return_value = mock_instance

            indicators = generator.generate_indicators()

            self.assertIsInstance(indicators, list)
            mock_random_class.assert_called_once_with(self.config)

    @patch('app.services.auto_strategy.generators.condition_generator.ConditionGenerator')
    def test_generate_conditions_smart(self, mock_condition_gen_class):
        """スマート条件生成テスト"""
        generator = SmartGeneGenerator(self.config)
        indicators = [IndicatorGene(type="SMA", enabled=True)]

        mock_condition_gen = MagicMock()
        mock_condition_gen.generate_balanced_conditions.return_value = (
            [MagicMock()],  # long_conditions
            [MagicMock()],  # short_conditions
            []  # exit_conditions
        )
        mock_condition_gen_class.return_value = mock_condition_gen

        with patch.object(generator, 'generate_indicators', return_value=indicators):
            entry_conditions, long_conditions, short_conditions = (
                generator.generate_conditions(indicators)
            )

        self.assertIsInstance(entry_conditions, list)
        self.assertIsInstance(long_conditions, list)
        self.assertIsInstance(short_conditions, list)

    def test_generate_conditions_fallback(self):
        """条件生成フォールバックテスト"""
        generator = SmartGeneGenerator(self.config)
        indicators = [IndicatorGene(type="SMA", enabled=True)]

        with patch('app.services.auto_strategy.generators.condition_generator.ConditionGenerator') as mock_condition_gen_class:
            mock_condition_gen_class.side_effect = Exception("Smart generation failed")

            entry_conditions, long_conditions, short_conditions = (
                generator.generate_conditions(indicators)
            )

        # フォールバック条件を確認
        self.assertIsInstance(entry_conditions, list)
        self.assertIsInstance(long_conditions, list)
        self.assertIsInstance(short_conditions, list)

    def test_generate_tpsl_gene(self):
        """TP/SL遺伝子生成テスト"""
        generator = SmartGeneGenerator(self.config)

        tpsl_gene = generator.generate_tpsl_gene()

        self.assertIsInstance(tpsl_gene, TPSLGene)

    def test_generate_position_sizing_gene(self):
        """ポジションサイジング遺伝子生成テスト"""
        generator = SmartGeneGenerator(self.config)

        position_sizing_gene = generator.generate_position_sizing_gene()

        self.assertIsInstance(position_sizing_gene, PositionSizingGene)


class TestDefaultGeneGenerator(unittest.TestCase):

    def setUp(self):
        self.config = MockConfig()

    def test_init(self):
        """初期化テスト"""
        generator = DefaultGeneGenerator(self.config)
        self.assertEqual(generator.config, self.config)

    def test_generate_indicators(self):
        """デフォルト指標生成テスト"""
        generator = DefaultGeneGenerator(self.config)

        indicators = generator.generate_indicators()

        self.assertIsInstance(indicators, list)
        self.assertEqual(len(indicators), 2)
        types = [ind.type for ind in indicators]
        self.assertIn("SMA", types)
        self.assertIn("RSI", types)

    def test_generate_conditions(self):
        """条件生成テスト"""
        generator = DefaultGeneGenerator(self.config)
        indicators = [IndicatorGene(type="SMA", enabled=True)]

        entry_conditions, long_conditions, short_conditions = (
            generator.generate_conditions(indicators)
        )

        self.assertIsInstance(entry_conditions, list)
        self.assertIsInstance(long_conditions, list)
        self.assertIsInstance(short_conditions, list)
        self.assertGreater(len(entry_conditions), 0)
        self.assertGreater(len(long_conditions), 0)
        self.assertGreater(len(short_conditions), 0)


class TestBaseGeneGenerator(unittest.TestCase):

    def setUp(self):
        self.config = MockConfig()
        self.generator = DefaultGeneGenerator(self.config)  # BaseをテストするためDefaultを使用

    def test_generate_tpsl_gene(self):
        """TP/SL遺伝子生成テスト"""
        tpsl_gene = self.generator.generate_tpsl_gene()
        self.assertIsInstance(tpsl_gene, TPSLGene)

    def test_generate_position_sizing_gene(self):
        """ポジションサイジング遺伝子生成テスト"""
        position_sizing_gene = self.generator.generate_position_sizing_gene()
        self.assertIsInstance(position_sizing_gene, PositionSizingGene)

    def test_generate_risk_management(self):
        """リスク管理設定生成テスト"""
        risk_management = self.generator.generate_risk_management()
        self.assertIsInstance(risk_management, dict)
        self.assertIn("position_size", risk_management)

    @patch('app.services.auto_strategy.models.strategy_models.create_random_tpsl_gene')
    @patch('app.services.auto_strategy.models.strategy_models.create_random_position_sizing_gene')
    def test_generate_complete_gene_success(
        self,
        mock_create_position_sizing,
        mock_create_tpsl
    ):
        """完全な遺伝子生成テスト（成功時）"""
        mock_tpsl = MagicMock(spec=TPSLGene)
        mock_position_sizing = MagicMock(spec=PositionSizingGene)

        mock_create_tpsl.return_value = mock_tpsl
        mock_create_position_sizing.return_value = mock_position_sizing

        gene = self.generator.generate_complete_gene()

        self.assertIsInstance(gene, StrategyGene)
        self.assertIsInstance(gene.indicators, list)
        self.assertIsInstance(gene.entry_conditions, list)
        self.assertIsInstance(gene.long_entry_conditions, list)
        self.assertIsInstance(gene.short_entry_conditions, list)

    def test_generate_complete_gene_fallback(self):
        """完全な遺伝子生成フォールバックテスト"""
        with patch.object(self.generator, 'generate_indicators', side_effect=Exception("Test error")):
            gene = self.generator.generate_complete_gene()

        # フォールバック遺伝子が返されるべき
        self.assertIsInstance(gene, StrategyGene)
        self.assertIn("Fallback", gene.metadata.get("generated_by", ""))

    def test_create_fallback_gene(self):
        """フォールバック遺伝子作成テスト"""
        generator = DefaultGeneGenerator(self.config)
        fallback_gene = generator._create_fallback_gene()

        self.assertIsInstance(fallback_gene, StrategyGene)
        self.assertGreater(len(fallback_gene.indicators), 0)
        self.assertIsInstance(fallback_gene.tpsl_gene, TPSLGene)
        self.assertIsInstance(fallback_gene.position_sizing_gene, PositionSizingGene)


if __name__ == '__main__':
    unittest.main()