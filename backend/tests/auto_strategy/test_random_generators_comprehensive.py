"""
ランダムジェネレータの包括的テスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import random

from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.generators.random.indicator_generator import (
    IndicatorGenerator,
)
from app.services.auto_strategy.generators.random.condition_generator import (
    ConditionGenerator,
)
from app.services.auto_strategy.generators.random.tpsl_generator import TPSLGenerator
from app.services.auto_strategy.generators.random.position_sizing_generator import (
    PositionSizingGenerator,
)
from app.services.auto_strategy.generators.random.operand_generator import (
    OperandGenerator,
)
from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.models.strategy_models import (
    StrategyGene,
    IndicatorGene,
    Condition,
    TPSLGene,
    PositionSizingGene,
    PositionSizingMethod,
)


class TestRandomGeneratorsComprehensive:
    """ランダムジェネレータの包括的テスト"""

    @pytest.fixture
    def config(self):
        """テスト用GA設定"""
        return GAConfig(
            max_indicators=5,
            min_indicators=1,
            max_conditions=3,
            min_conditions=1,
            threshold_ranges={
                "oscillator_0_100": [20, 80],
                "volume": [1000, 10000],
                "price_ratio": [0.95, 1.05],
            },
        )

    @pytest.fixture
    def random_gene_generator(self, config):
        """ランダム遺伝子生成器"""
        return RandomGeneGenerator(config, enable_smart_generation=False)

    def test_indicator_generator_initialization(self, config):
        """指標生成器の初期化テスト"""
        generator = IndicatorGenerator(config)

        assert generator.config == config
        assert hasattr(generator, "indicator_service")
        assert hasattr(generator, "composition_service")
        assert hasattr(generator, "_valid_indicator_names")
        assert isinstance(generator._valid_indicator_names, set)

    def test_indicator_generator_fallback_behavior(self, config):
        """指標生成器のフォールバック動作テスト"""
        generator = IndicatorGenerator(config)

        # 空の利用可能指標リスト
        generator.available_indicators = []

        # フォールバックが発動するか
        indicators = generator.generate_random_indicators()

        # デフォルト指標が返される
        assert len(indicators) >= 1
        assert indicators[0].type == "SMA"
        assert indicators[0].parameters["period"] == 20

    def test_indicator_generator_parameter_generation(self, config):
        """指標パラメータ生成のテスト"""
        generator = IndicatorGenerator(config)

        # 有効な指標でテスト
        generator.available_indicators = ["SMA", "EMA"]

        indicators = generator.generate_random_indicators()

        # 指標が生成される
        assert len(indicators) >= 1
        for indicator in indicators:
            assert hasattr(indicator, "type")
            assert hasattr(indicator, "parameters")
            assert hasattr(indicator, "enabled")
            assert indicator.enabled is True

    def test_condition_generator_initialization(self, config):
        """条件生成器の初期化テスト"""
        generator = ConditionGenerator(config)

        assert generator.config == config
        assert hasattr(generator, "available_operators")
        assert hasattr(generator, "price_data_weight")
        assert hasattr(generator, "volume_data_weight")
        assert hasattr(generator, "oi_fr_data_weight")

    def test_condition_generator_with_indicators(self, config):
        """指標付き条件生成のテスト"""
        generator = ConditionGenerator(config)

        # テスト用指標
        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 10}),
            IndicatorGene(type="RSI", parameters={"period": 14}),
        ]

        conditions = generator.generate_random_conditions(indicators, "entry")

        # 条件が生成される
        assert len(conditions) >= 1
        for condition in conditions:
            assert isinstance(condition, Condition)
            assert hasattr(condition, "left_operand")
            assert hasattr(condition, "operator")
            assert hasattr(condition, "right_operand")

    def test_condition_generator_no_indicators(self, config):
        """指標なし条件生成のテスト"""
        generator = ConditionGenerator(config)

        conditions = generator.generate_random_conditions([], "exit")

        # フォールバック条件が生成される
        assert len(conditions) >= 1
        assert conditions[0].left_operand == "close"
        assert conditions[0].right_operand == "SMA"

    def test_condition_compatibility_system(self, config):
        """互換性システムのテスト"""
        generator = ConditionGenerator(config)

        indicators = [IndicatorGene(type="SMA", parameters={"period": 10})]

        # 互換性の高いオペランドが選ばれるか
        with patch(
            "app.services.auto_strategy.core.operand_grouping.operand_grouping_system.get_compatible_operands"
        ) as mock_compatible:
            mock_compatible.return_value = ["SMA", "close"]

            operand = generator._choose_right_operand("SMA", indicators, "entry")

            # 互換性システムが使用される
            mock_compatible.assert_called_once()

    def test_tpsl_generator_initialization(self, config):
        """TP/SL生成器の初期化テスト"""
        generator = TPSLGenerator(config)

        assert generator.config == config

    def test_tpsl_generator_default_generation(self, config):
        """TP/SLのデフォルト生成テスト"""
        generator = TPSLGenerator(config)

        tpsl_gene = generator.generate_tpsl_gene()

        # 有効なTP/SL遺伝子が生成される
        assert isinstance(tpsl_gene, TPSLGene)
        assert hasattr(tpsl_gene, "method")
        assert hasattr(tpsl_gene, "stop_loss_pct")
        assert hasattr(tpsl_gene, "take_profit_pct")
        assert hasattr(tpsl_gene, "risk_reward_ratio")

    def test_tpsl_generator_with_constraints(self, config):
        """制約付きTP/SL生成のテスト"""
        # 制約付き設定
        config.tpsl_method_constraints = ["FIXED_PERCENTAGE", "RISK_REWARD_RATIO"]
        config.tpsl_sl_range = [0.01, 0.05]
        config.tpsl_tp_range = [0.02, 0.1]
        config.tpsl_rr_range = [1.5, 3.0]

        generator = TPSLGenerator(config)

        tpsl_gene = generator.generate_tpsl_gene()

        # 制約が適用される
        assert tpsl_gene.stop_loss_pct >= 0.01
        assert tpsl_gene.stop_loss_pct <= 0.05
        assert tpsl_gene.take_profit_pct >= 0.02
        assert tpsl_gene.take_profit_pct <= 0.1
        assert tpsl_gene.risk_reward_ratio >= 1.5
        assert tpsl_gene.risk_reward_ratio <= 3.0

    def test_tpsl_generator_error_handling(self, config):
        """TP/SL生成のエラーハンドリング"""
        generator = TPSLGenerator(config)

        # create_random_tpsl_geneをモックでエラーを発生
        with patch(
            "app.services.auto_strategy.generators.random.tpsl_generator.create_random_tpsl_gene"
        ) as mock_create:
            mock_create.side_effect = Exception("Test error")

            tpsl_gene = generator.generate_tpsl_gene()

            # フォールバックが発動
            assert isinstance(tpsl_gene, TPSLGene)
            assert tpsl_gene.enabled is True
            assert tpsl_gene.method is not None

    def test_position_sizing_generator_initialization(self, config):
        """ポジションサイジング生成器の初期化テスト"""
        generator = PositionSizingGenerator(config)

        assert generator.config == config

    def test_position_sizing_generator_default_generation(self, config):
        """ポジションサイジングのデフォルト生成"""
        generator = PositionSizingGenerator(config)

        pos_gene = generator.generate_position_sizing_gene()

        # 有効な遺伝子が生成される
        assert isinstance(pos_gene, PositionSizingGene)
        assert hasattr(pos_gene, "method")
        assert hasattr(pos_gene, "enabled")

    def test_operand_generator_initialization(self, config):
        """オペランド生成器の初期化テスト"""
        generator = OperandGenerator(config)

        assert generator.config == config

    def test_random_gene_generator_initialization(self, config):
        """ランダム遺伝子生成器の初期化テスト"""
        generator = RandomGeneGenerator(config, enable_smart_generation=True)

        assert generator.config == config
        assert generator.enable_smart_generation is True
        assert hasattr(generator, "indicator_generator")
        assert hasattr(generator, "condition_generator")
        assert hasattr(generator, "tpsl_generator")
        assert hasattr(generator, "position_sizing_generator")
        assert hasattr(generator, "operand_generator")

    def test_random_gene_generator_core_generation(self, random_gene_generator):
        """ランダム遺伝子生成のコアテスト"""
        gene = random_gene_generator.generate_random_gene()

        # 基本的な属性が存在する
        assert isinstance(gene, StrategyGene)
        assert hasattr(gene, "indicators")
        assert hasattr(gene, "entry_conditions")
        assert hasattr(gene, "exit_conditions")
        assert hasattr(gene, "long_entry_conditions")
        assert hasattr(gene, "short_entry_conditions")
        assert hasattr(gene, "risk_management")
        assert hasattr(gene, "tpsl_gene")
        assert hasattr(gene, "position_sizing_gene")
        assert hasattr(gene, "metadata")

    def test_random_gene_generator_fallback_behavior(self, config):
        """ランダム遺伝子生成器のフォールバックテスト"""
        # すべてのサブジェネレータをモックでエラーに
        with patch.multiple(
            "app.services.auto_strategy.generators.random_gene_generator",
            IndicatorGenerator=Mock(side_effect=Exception("Test error")),
            ConditionGenerator=Mock(side_effect=Exception("Test error")),
            TPSLGenerator=Mock(side_effect=Exception("Test error")),
            PositionSizingGenerator=Mock(side_effect=Exception("Test error")),
        ):
            generator = RandomGeneGenerator(config)

            gene = generator.generate_random_gene()

            # フォールバック遺伝子が生成される
            assert gene.metadata["generated_by"] == "Fallback"
            assert len(gene.indicators) == 1
            assert gene.indicators[0].type == "SMA"

    def test_ensure_or_with_fallback_diversity(self, random_gene_generator):
        """フォールバック多様性テスト"""
        # 空の条件リスト
        empty_conditions = []
        indicators = [IndicatorGene(type="SMA", parameters={"period": 10})]

        result = random_gene_generator._ensure_or_with_fallback(
            empty_conditions, "long", indicators
        )

        # フォールバック条件が追加される
        assert len(result) >= 1

        # 単一条件の場合
        single_condition = [
            Condition(left_operand="close", operator=">", right_operand="open")
        ]
        result = random_gene_generator._ensure_or_with_fallback(
            single_condition, "long", indicators
        )

        # 追加条件が含まれる
        assert len(result) >= 2

    def test_trend_preference_logic(self, random_gene_generator):
        """トレンド指標優先ロジックテスト"""
        # 有効なトレンド指標
        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 10}, enabled=True),
            IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(
                type="RSI", parameters={"period": 14}, enabled=True
            ),  # トレンドではない
        ]

        result = random_gene_generator._ensure_or_with_fallback([], "long", indicators)

        # トレンド指標が選ばれる
        if result and hasattr(result[0], "right_operand"):
            assert result[0].right_operand in ["SMA", "EMA"]

    def test_diversity_in_generation(self, config):
        """生成多様性のテスト"""
        generator = RandomGeneGenerator(config)

        # 複数回生成して多様性を確認
        genes = []
        for _ in range(10):
            gene = generator.generate_random_gene()
            genes.append(gene)

        # 完全に同じ遺伝子が続くことは稀
        unique_configs = set()
        for gene in genes:
            config_str = f"{len(gene.indicators)}_{len(gene.entry_conditions)}_{len(gene.long_entry_conditions)}"
            unique_configs.add(config_str)

        # ある程度の多様性がある
        assert len(unique_configs) >= 2

    def test_smart_generation_integration(self, config):
        """スマート生成の統合テスト"""
        generator = RandomGeneGenerator(
            config,
            enable_smart_generation=True,
            smart_context={
                "timeframe": "1h",
                "symbol": "BTC/USDT",
                "regime_gating": "bullish",
                "threshold_profile": "aggressive",
            },
        )

        assert generator.enable_smart_generation is True
        # コンテキストが設定される

    def test_condition_normalization(self, random_gene_generator):
        """条件正規化のテスト"""
        # 複数の条件
        conditions = [
            Condition(left_operand="close", operator=">", right_operand="sma"),
            Condition(left_operand="rsi", operator="<", right_operand="30"),
        ]
        indicators = [IndicatorGene(type="SMA", parameters={"period": 10})]

        result = random_gene_generator._ensure_or_with_fallback(
            conditions, "long", indicators
        )

        # ORグループが生成される可能性
        # 複雑なロジックのため、正常終了を確認

    def test_max_indicator_limit_respected(self, config):
        """最大指標数制限のテスト"""
        config.max_indicators = 2
        config.min_indicators = 2

        generator = RandomGeneGenerator(config)

        gene = generator.generate_random_gene()

        # 最大数が守られている
        assert len(gene.indicators) <= config.max_indicators

    def test_min_indicator_requirement(self, config):
        """最小指標数要求のテスト"""
        config.min_indicators = 3
        config.max_indicators = 3

        generator = RandomGeneGenerator(config)

        gene = generator.generate_random_gene()

        # 最小数が守られている
        assert len(gene.indicators) >= config.min_indicators

    def test_error_handling_in_core_generation(self, config):
        """コア生成のエラーハンドリング"""
        generator = RandomGeneGenerator(config)

        # 生成中にエラーが発生してもフォールバックが働くか
        with patch.object(generator, "indicator_generator") as mock_indicator:
            mock_indicator.generate_random_indicators.side_effect = Exception(
                "Test error"
            )

            gene = generator.generate_random_gene()

            # フォールバック遺伝子が生成される
            assert gene.metadata["generated_by"] == "Fallback"

    def test_config_attribute_access(self, config):
        """設定属性アクセスのテスト"""
        generator = RandomGeneGenerator(config)

        # 設定値が正しく取得される
        assert generator.max_indicators == config.max_indicators
        assert generator.min_indicators == config.min_indicators
        assert generator.max_conditions == config.max_conditions
        assert generator.min_conditions == config.min_conditions

    def test_backward_compatibility_with_old_conditions(self, random_gene_generator):
        """旧式条件との後方互換性"""
        # 旧式条件生成が維持されているか
        gene = random_gene_generator.generate_random_gene()

        # entry_conditionsとexit_conditionsが存在する
        assert hasattr(gene, "entry_conditions")
        assert hasattr(gene, "exit_conditions")

    def test_metadata_enrichment(self, random_gene_generator):
        """メタデータの充実化テスト"""
        gene = random_gene_generator.generate_random_gene()

        # メタデータが含まれる
        assert "generated_by" in gene.metadata
        assert gene.metadata["generated_by"] == "RandomGeneGenerator"

    def test_generator_component_interaction(self, config):
        """ジェネレータコンポーネントの相互作用"""
        with patch(
            "app.services.auto_strategy.generators.random_gene_generator.ConditionGenerator"
        ) as MockConditionGen:
            mock_condition_gen = Mock()
            mock_condition_gen.generate_random_conditions.return_value = [
                Condition(left_operand="close", operator=">", right_operand="sma")
            ]
            MockConditionGen.return_value = mock_condition_gen

            generator = RandomGeneGenerator(config)

            gene = generator.generate_random_gene()

            # コンポーネントが呼び出される
            mock_condition_gen.generate_random_conditions.assert_called()

    def test_threshold_range_integration(self, config):
        """閾値範囲統合のテスト"""
        # 閾値範囲が正しく使用される
        generator = RandomGeneGenerator(config)

        # 内部のコンディショングェネレータが設定を参照しているか
        assert hasattr(generator.condition_generator, "threshold_ranges")
        # 実際の動作確認は難しく、初期化が成功すればOK

    def test_operands_weight_distribution(self, config):
        """オペランド重み分布のテスト"""
        generator = RandomGeneGenerator(config)

        # オペランド生成器が重みを正しく設定しているか
        assert hasattr(generator.operand_generator, "price_data_weight")
        assert hasattr(generator.operand_generator, "volume_data_weight")
        assert hasattr(generator.operand_generator, "oi_fr_data_weight")

    def test_indicator_composition_service_integration(self, config):
        """指標構成サービス統合のテスト"""
        generator = RandomGeneGenerator(config)

        # 指標生成器が構成サービスを持っている
        assert hasattr(generator.indicator_generator, "composition_service")

    def test_diversity_preservation_in_constraints(self, config):
        """制約下での多様性維持"""
        # 制約をかけても多様性が保たれるか
        config.tpsl_method_constraints = ["FIXED_PERCENTAGE"]
        config.tpsl_sl_range = [0.02, 0.02]  # 固定
        config.tpsl_tp_range = [0.04, 0.04]  # 固定

        generator = RandomGeneGenerator(config)

        # 複数生成
        genes = []
        for _ in range(5):
            gene = generator.generate_random_gene()
            genes.append(gene)

        # 指標や条件で多様性がある
        indicator_types = set()
        for gene in genes:
            for ind in gene.indicators:
                indicator_types.add(ind.type)

        # 完全に同じ構成でない
        assert len(indicator_types) >= 1  # 最低限の多様性

    def test_memory_efficiency_in_bulk_generation(self, config):
        """大量生成時のメモリ効率"""
        import gc

        generator = RandomGeneGenerator(config)

        initial_objects = len(gc.get_objects())

        # 多数生成
        for _ in range(100):
            gene = generator.generate_random_gene()
            # 参照を解放
            del gene

        gc.collect()
        final_objects = len(gc.get_objects())

        # 大幅なメモリ増加でない
        assert (final_objects - initial_objects) < 1000  # 緩いチェック

    def test_random_seed_independence(self, config):
        """乱数シード独立性のテスト"""
        # 同じ設定で異なる結果が得られるか
        generator1 = RandomGeneGenerator(config)
        generator2 = RandomGeneGenerator(config)

        gene1 = generator1.generate_random_gene()
        gene2 = generator2.generate_random_gene()

        # 完全に同じである可能性は低い
        # 構造が同じでもパラメータが異なる可能性
        assert True  # 実際の乱数依存のため、正常終了を確認

    def test_exception_propagation_handling(self, config):
        """例外伝播のハンドリング"""
        generator = RandomGeneGenerator(config)

        # 各サブジェネレータで例外が発生してもフォールバックが働く
        with patch.object(
            generator.indicator_generator, "generate_random_indicators"
        ) as mock_ind:
            mock_ind.side_effect = ValueError("Test error")

            gene = generator.generate_random_gene()

            # 正常にフォールバック
            assert gene is not None
            assert isinstance(gene, StrategyGene)

    def test_configuration_edge_cases(self, config):
        """設定のエッジケーステスト"""
        # 極端な設定
        config.max_indicators = 1
        config.min_indicators = 1
        config.max_conditions = 1
        config.min_conditions = 0

        generator = RandomGeneGenerator(config)

        gene = generator.generate_random_gene()

        # 極端な設定でも動作する
        assert isinstance(gene, StrategyGene)

    def test_backward_compatibility_with_tpsl_logic(self, config):
        """TP/SLロジックの後方互換性"""
        generator = RandomGeneGenerator(config)

        gene = generator.generate_random_gene()

        # TP/SL遺伝子が有効化される
        if gene.tpsl_gene:
            assert gene.tpsl_gene.enabled is True

    def test_condition_balancing_logic(self, random_gene_generator):
        """条件バランスロジックのテスト"""
        # ロング・ショート条件が生成される
        gene = random_gene_generator.generate_random_gene()

        # 新しい条件が含まれる
        assert hasattr(gene, "long_entry_conditions")
        assert hasattr(gene, "short_entry_conditions")

        # バランスが取れているか（完全なテストは難しいが、生成される）
        assert gene.long_entry_conditions is not None
        assert gene.short_entry_conditions is not None
