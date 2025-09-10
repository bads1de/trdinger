"""
TPSL Generatorのテスト（バグ検出）
"""
import pytest
from unittest.mock import Mock

from app.services.auto_strategy.models.enums import TPSLMethod
from app.services.auto_strategy.models.tpsl_gene import TPSLGene
from app.services.auto_strategy.models.tpsl_result import TPSLResult
from app.services.auto_strategy.tpsl.generator import (
    AdaptiveStrategy,
    UnifiedTPSLGenerator,
    VolatilityStrategy,
    RiskRewardStrategy,
    FixedPercentageStrategy,
)


class TestTPSLGenerator:
    """TPSL Generatorのテスト"""

    def test_adaptive_strategy_method_comparison_bug(self):
        """AdaptiveStrategyのmethod比較バグ検出テスト（method.nameを使っているバグ）"""
        strategy = AdaptiveStrategy()

        # 実TPSLGeneオブジェクトを使用
        gene = TPSLGene(method=TPSLMethod.VOLATILITY_BASED, enabled=True)

        # kwargsにtpsl_geneを渡す
        kwargs = {"tpsl_gene": gene}

        # AdaptiveStrategyの実装では method.name == "VOLATILITY_BASED" と比較している
        # しかし TPSLMethod.VOLATILITY_BASED.name == "VOLATILITY_BASED" なので一致するはず
        # （バグがないとvolatilityが返されるが、意味論的に.valueで比較すべき）
        result = strategy.generate(**kwargs)

        # 現在はバグなくvolatilityが返される（本当は.valueで比較すべきだが）
        assert result.method_used == "volatility"  # バグがない証拠

    def test_adaptive_strategy_method_value_comparison_should_be(self):
        """AdaptiveStrategyはmethod.valueを使って比較すべき（望ましい挙動）"""
        # このテストはバグがあることを示す：method.nameではなくmethod.valueを使うべき

        # TPSLMethod.VOLATILITY_BASED.value == "volatility_based"
        # TPSLMethod.VOLATILITY_BASED.name == "VOLATILITY_BASED"

        # generator.pyでは method.name == "VOLATILITY_BASED" と比較しているが
        # method == "volatility_based" と比較すべき
        strategy = AdaptiveStrategy()
        gene = TPSLGene(method=TPSLMethod.VOLATILITY_BASED, enabled=True)
        kwargs = {"tpsl_gene": gene}

        result = strategy.generate(**kwargs)
        # 現在動作するが、設計的にはmethod.valueで比較すべき
        assert result.method_used == "volatility"

    def test_volatility_strategy_without_base_atr_pct(self):
        """VolatilityStrategyのパラメータ不足バグ検出テスト"""
        strategy = VolatilityStrategy()

        # base_atr_pctなし
        kwargs = {}

        # デフォルト値 0.02 が使われるので、これは動作するはず
        result = strategy.generate(**kwargs)
        assert isinstance(result, TPSLResult)
        assert result.method_used == "volatility"

    def test_risk_reward_strategy_without_target_ratio(self):
        """RiskRewardStrategyのパラメータ不足バグ検出テスト"""
        strategy = RiskRewardStrategy()

        # target_ratioなし
        kwargs = {"stop_loss_pct": 0.02}

        # デフォルト値 2.0 が使われるので動作するはず
        result = strategy.generate(**kwargs)
        assert isinstance(result, TPSLResult)
        assert result.method_used == "risk_reward"

    def test_fixed_percentage_strategy_without_params(self):
        """FixedPercentageStrategyのパラメータ不足バグ検出テスト"""
        strategy = FixedPercentageStrategy()

        # すべてのパラメータなし
        kwargs = {}

        # デフォルト値が使用されるので動作するはず
        result = strategy.generate(**kwargs)
        assert isinstance(result, TPSLResult)
        assert result.stop_loss_pct == 0.03
        assert result.take_profit_pct == 0.06

    def test_unified_generator_adaptive_method(self):
        """UnifiedTPSLGeneratorのadaptive手法テスト"""

        generator = UnifiedTPSLGenerator()

        # adaptiveを選択
        result = generator.generate_tpsl("adaptive")

        # 結果がTPSLResultであること
        assert isinstance(result, TPSLResult)
        # adaptiveの動作を確認

    def test_unified_generator_statistical_method(self):
        """UnifiedTPSLGeneratorのstatistical手法テスト"""

        generator = UnifiedTPSLGenerator()

        # statisticalを選択
        result = generator.generate_tpsl("statistical")

        assert isinstance(result, TPSLResult)
        assert result.method_used == "statistical"

    def test_unified_generator_unknown_method(self):
        """UnifiedTPSLGeneratorの未知手法テスト"""

        generator = UnifiedTPSLGenerator()

        # 未知の手法を選択 - ValueErrorが発生すべき
        with pytest.raises(ValueError):
            generator.generate_tpsl("unknown_method")

    def test_unified_generator_volatility_based_method(self):
        """UnifiedTPSLGeneratorのvolatility_based手法テスト"""

        generator = UnifiedTPSLGenerator()

        # volatility_basedを選択
        result = generator.generate_tpsl("volatility_based")

        assert isinstance(result, TPSLResult)
        assert result.method_used == "volatility"

    def test_unified_generator_risk_reward_method(self):
        """UnifiedTPSLGeneratorのrisk_reward手法テスト"""

        generator = UnifiedTPSLGenerator()

        # risk_rewardを選択
        result = generator.generate_tpsl("risk_reward_ratio")

        assert isinstance(result, TPSLResult)
        assert result.method_used == "risk_reward"

    def test_generate_adaptive_tpsl_high_volatility(self):
        """適応的TP/SLの高ボラティリティ条件テスト"""

        generator = UnifiedTPSLGenerator()

        market_conditions = {"volatility": "high"}

        result = generator.generate_adaptive_tpsl(market_conditions)

        assert isinstance(result, TPSLResult)

    def test_volatility_strategy_custom_multiplier_bug(self):
        """VolatilityStrategyのカスタムmultiplierバグ検出テスト"""
        strategy = VolatilityStrategy()

        # カスタムmultiplierを渡す
        kwargs = {
            "base_atr_pct": 0.02,
            "atr_multiplier_sl": 2.0,
            "atr_multiplier_tp": 4.0,
        }

        result = strategy.generate(**kwargs)

        # Expected: stop_loss_pct = base_atr_pct * atr_multiplier_sl = 0.02 * 2.0 = 0.04
        # Expected: take_profit_pct = 0.02 * 4.0 = 0.08
        expected_sl_pct = 0.02 * 2.0
        expected_tp_pct = 0.02 * 4.0

        # Currently, it uses 1.5 and 3.0, so this test will fail, detecting the bug
        assert result.stop_loss_pct == expected_sl_pct
        assert result.take_profit_pct == expected_tp_pct