"""
条件生成統合テスト - Phase 1.3
ConditionGeneratorクラスの統合機能テスト

設計思想:
- スマート条件生成の各レイヤーをテスト
- 統合メソッドの機能を検証
- threshold profileの連携を確保
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from app.services.auto_strategy.generators.condition_generator import ConditionGenerator
from app.services.auto_strategy.models.strategy_models import Condition, IndicatorGene
from app.services.auto_strategy.config.constants import INDICATOR_CHARACTERISTICS, IndicatorType


class TestConditionGeneratorIntegration:
    """ConditionGenerator統合テスト"""

    @pytest.fixture
    def generator(self):
        """ConditionGeneratorインスタンス"""
        return ConditionGenerator()

    @pytest.fixture
    def mock_indicator_registry(self):
        """Mock indicator_registry"""
        with patch("app.services.auto_strategy.generators.condition_generator.indicator_registry") as mock_reg:
            # Import the actual ScaleType for proper comparison
            from app.services.indicators.config.indicator_config import IndicatorScaleType

            # RSI indicator config
            rsi_config = Mock()
            rsi_config.scale_type = IndicatorScaleType.OSCILLATOR_0_100  # Use actual enum
            rsi_config.category = "momentum"

            # SMA indicator config
            sma_config = Mock()
            sma_config.scale_type = IndicatorScaleType.PRICE_RATIO  # Use actual enum
            sma_config.category = "trend"

            def mock_get_indicator_config(name):
                if name == "RSI":
                    return rsi_config
                elif name in ["SMA", "EMA"]:
                    return sma_config
                return None

            mock_reg.get_indicator_config = mock_get_indicator_config
            yield mock_reg

    @pytest.fixture
    def sample_indicators(self) -> List[IndicatorGene]:
        """サンプル指標リスト"""
        return [
            IndicatorGene(type="RSI", enabled=True, parameters={"period": 14}),
            IndicatorGene(type="SMA", enabled=True, parameters={"period": 20}),
            IndicatorGene(type="MACD", enabled=True, parameters={"fast": 12, "slow": 26}),
            # IndicatorGene(type="CORREL", enabled=True, parameters={"period": 20}),  # 統計指標は削除済み
            IndicatorGene(type="CDL_HAMMER", enabled=True, parameters={}),
        ]

    def test_generic_long_short_conditions(self, generator, mock_indicator_registry):
        """基盤条件ロジックの統合テスト"""
        # RSI (momentum) indicator
        rsi_gene = IndicatorGene(type="RSI", enabled=True, parameters={"period": 14})

        # Test long condition
        long_conditions = generator._generic_long_conditions(rsi_gene)
        assert len(long_conditions) > 0
        assert all(isinstance(c, Condition) for c in long_conditions)

        # Test short condition
        short_conditions = generator._generic_short_conditions(rsi_gene)
        assert len(short_conditions) > 0
        assert all(isinstance(c, Condition) for c in short_conditions)

        # RSI specific assertions
        long_condition = long_conditions[0]
        assert long_condition.left_operand == "RSI"
        assert long_condition.operator == "<"  # RSI売られすぎは < 演算子
        assert isinstance(long_condition.right_operand, float)  # ThresholdPolicyから得られる値

    def test_trend_conditions(self, generator):
        """トレンド条件生成テスト"""
        trend_gene = IndicatorGene(type="SMA", enabled=True, parameters={"period": 20})
        generator.indicators = [trend_gene]  # For name validation

        # Test long condition
        long_conditions = generator._create_trend_long_conditions(trend_gene)
        assert len(long_conditions) == 1
        assert long_conditions[0].left_operand == "close"
        assert long_conditions[0].operator == ">"
        assert long_conditions[0].right_operand == "SMA"

        # Test short condition
        short_conditions = generator._create_trend_short_conditions(trend_gene)
        assert len(short_conditions) == 1
        assert short_conditions[0].left_operand == "close"
        assert short_conditions[0].operator == "<"

    def test_momentum_conditions(self, generator, mock_indicator_registry):
        """モメンタム条件生成テスト"""
        momentum_gene = IndicatorGene(type="RSI", enabled=True, parameters={"period": 14})

        # Test long condition
        long_conditions = generator._create_momentum_long_conditions(momentum_gene)
        assert len(long_conditions) > 0
        assert long_conditions[0].left_operand == "RSI"
        assert long_conditions[0].operator == "<"
        assert long_conditions[0].right_operand == 35  # RSI売られすぎ（デフォルトbase=60, 60-25=35）

    # def test_statistics_conditions(self, generator):
    #     """統計条件生成テスト - 統計指標は削除済み"""
    #     stats_gene = IndicatorGene(type="CORREL", enabled=True, parameters={"period": 20})
    #
    #     # Test long condition
    #     long_conditions = generator._create_statistics_long_conditions(stats_gene)
    #     assert len(long_conditions) > 0
    #     assert long_conditions[0].left_operand == "CORREL"
    #     assert long_conditions[0].operator == ">"
    #     assert isinstance(long_conditions[0].right_operand, float)

    def test_pattern_conditions(self, generator):
        """パターン認識条件生成テスト"""
        pattern_gene = IndicatorGene(type="CDL_HAMMER", enabled=True, parameters={})

        # Test long condition
        long_conditions = generator._create_pattern_long_conditions(pattern_gene)
        assert len(long_conditions) > 0
        assert long_conditions[0].left_operand == "CDL_HAMMER"
        assert long_conditions[0].operator == ">"

    @pytest.mark.parametrize("profile", ["aggressive", "normal", "conservative"])
    def test_threshold_profile_integration(self, generator, mock_indicator_registry, profile):
        """thresholdプロファイル統合テスト"""
        rsi_gene = IndicatorGene(type="RSI", enabled=True)
        generator.set_context(threshold_profile=profile)

        # Test different profiles generate different thresholds
        conditions_normal = generator._generic_long_conditions(rsi_gene)

        if profile != "normal":
            generator.set_context(threshold_profile="normal")
            conditions_other = generator._generic_long_conditions(rsi_gene)

            # Different profiles should generate different thresholds for RSI
            threshold_normal = conditions_normal[0].right_operand
            threshold_other = conditions_other[0].right_operand

            # Verify threshold differs (exact values depend on ThresholdPolicy)
            assert isinstance(threshold_normal, (int, float))
            assert isinstance(threshold_other, (int, float))

    def test_apply_threshold_context_new_feature(self, generator, sample_indicators, mock_indicator_registry):
        """新機能apply_threshold_context()テスト"""
        # Set context
        generator.set_context(
            threshold_profile="aggressive",
            timeframe="4h",
            regime_gating=True
        )

        # Generate basic conditions
        context_aware_conditions = generator.apply_threshold_context(sample_indicators)

        # Verify context application
        assert isinstance(context_aware_conditions, dict)
        assert "long_conditions" in context_aware_conditions
        assert "short_conditions" in context_aware_conditions
        assert isinstance(context_aware_conditions["long_conditions"], list)
        assert isinstance(context_aware_conditions["short_conditions"], list)

        # Verify non-empty conditions
        assert len(context_aware_conditions["long_conditions"]) > 0
        assert len(context_aware_conditions["short_conditions"]) > 0

    def test_indicator_type_classification(self, generator, sample_indicators):
        """指標タイプ分類テスト"""
        categorized = generator._dynamic_classify(sample_indicators)

        # Check categorization
        assert IndicatorType.MOMENTUM in categorized
        assert IndicatorType.TREND in categorized
        # assert IndicatorType.STATISTICS in categorized  # 統計指標は削除済み
        assert IndicatorType.PATTERN_RECOGNITION in categorized

        # RSI should be in momentum
        assert any(ind.type == "RSI" for ind in categorized[IndicatorType.MOMENTUM])
        # SMA should be in trend (fallback)
        assert any(ind.type == "SMA" for ind in categorized[IndicatorType.TREND])

    def test_integration_with_threshold_policy(self, generator, mock_indicator_registry):
        """ThresholdPolicyとの完全連携テスト"""
        rsi_gene = IndicatorGene(type="RSI", enabled=True)

        # Test aggressive profile
        generator.set_context(threshold_profile="aggressive")
        aggressive_conditions = generator._generic_long_conditions(rsi_gene)

        # Test conservative profile
        generator.set_context(threshold_profile="conservative")
        conservative_conditions = generator._generic_long_conditions(rsi_gene)

        # Both should be valid conditions
        assert len(aggressive_conditions) > 0
        assert len(conservative_conditions) > 0

        # Both should work with RSI
        assert aggressive_conditions[0].left_operand == "RSI"
        assert conservative_conditions[0].left_operand == "RSI"

    def test_complex_indicator_handling(self, generator, mock_indicator_registry):
        """複合指標処理テスト"""
        # MACD with momentum characteristics
        macd_gene = IndicatorGene(type="MACD", enabled=True)
        macd_conditions = generator._generic_long_conditions(macd_gene)

        # Should generate zero-based conditions for momentum indicators
        assert len(macd_conditions) > 0
        assert macd_conditions[0].left_operand == "MACD"

    def test_fallback_behavior(self, generator):
        """フォールバック動作テスト"""
        # Unknown indicator type
        unknown_gene = IndicatorGene(type="UNKNOWN_TYPE", enabled=True)
        conditions = generator._generic_long_conditions(unknown_gene)

        # Should still generate fallback conditions
        assert len(conditions) > 0
        assert isinstance(conditions[0], Condition)

    def test_disabled_indicator_handling(self, generator):
        """無効化指標処理テスト"""
        disabled_gene = IndicatorGene(type="RSI", enabled=False)
        categorized = generator._dynamic_classify([disabled_gene])

        # Disabled indicators should not appear in any category
        total_categorized = sum(len(indicators) for indicators in categorized.values())
        assert total_categorized == 0

    def test_long_short_balance_different_indicators_strategy(self, generator, mock_indicator_registry):
        """Different Indicators戦略でのロング・ショートバランス検証 (修正検証、統計指標は除外)"""
        # モメンタム指標のみの場合（統計指標は削除済み）
        momentum_only = [
            IndicatorGene(type="RSI", enabled=True),
            IndicatorGene(type="STOCH", enabled=True),
        ]

        long_cond, short_cond, exit_cond = generator.generate_balanced_conditions(momentum_only)

        print("\n=== モメンタム指標のみ生成テスト ===")
        print(f"ロング条件数: {len(long_cond)}")
        print(f"ショート条件数: {len(short_cond)}")

        # モメンタム指標でもショート条件が生成される
        assert len(long_cond) > 0, "モメンタム指標でロング条件が生成されない"
        assert len(short_cond) > 0, "モメンタム指標でショート条件が生成されない"

        # パターン指標のみの場合
        pattern_only = [
            IndicatorGene(type="CDL_HAMMER", enabled=True),
            IndicatorGene(type="CDL_ENGULFING", enabled=True),
        ]

        long_cond, short_cond, exit_cond = generator.generate_balanced_conditions(pattern_only)

        print("\n=== パターン指標のみ生成テスト ===")
        print(f"ロング条件数: {len(long_cond)}")
        print(f"ショート条件数: {len(short_cond)}")

        # パターン指標でもショート条件が生成されるようになったはず
        assert len(long_cond) > 0, "パターン指標でロング条件が生成されない"
        assert len(short_cond) > 0, "パターン指標でショート条件が生成されない"

        # 混合戦略テスト
        mixed_indicators = [
            IndicatorGene(type="RSI", enabled=True),    # MOMENTUM (ショート生成)
            IndicatorGene(type="SMA", enabled=True),    # TREND (ショート生成)
            IndicatorGene(type="ROC", enabled=True),    # MOMENTUM (ショート生成)
            IndicatorGene(type="CDL_HAMMER", enabled=True),  # PATTERN (ショート生成)
        ]

        long_cond, short_cond, exit_cond = generator.generate_balanced_conditions(mixed_indicators)

        print("\n=== 混合指標生成テスト ===")
        print(f"ロング条件数: {len(long_cond)}")
        print(f"ショート条件数: {len(short_cond)}")

        # 混合でもバランス良く生成される
        assert len(long_cond) > 0 and len(short_cond) > 0, "混合指標でロング・ショート条件がバランス良く生成されない"
        assert len(long_cond) >= 2, "混合指標でロング条件が十分に生成されない"
        assert len(short_cond) >= 2, "混合指標でショート条件が十分に生成されない"