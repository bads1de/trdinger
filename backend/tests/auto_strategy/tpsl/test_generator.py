"""
UnifiedTPSLGenerator Tests
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from app.services.auto_strategy.tpsl.generator import (
    UnifiedTPSLGenerator,
    RiskRewardStrategy,
    StatisticalStrategy,
    VolatilityStrategy,
    FixedPercentageStrategy,
    AdaptiveStrategy,
    TPSLMethod,
)


class TestTPSLStrategies:
    """個々のTPSL戦略クラスのテスト"""

    def test_risk_reward_strategy(self):
        strategy = RiskRewardStrategy()
        result = strategy.generate(stop_loss_pct=0.02, target_ratio=3.0)
        assert result.stop_loss_pct == 0.02
        assert result.take_profit_pct == 0.06
        assert result.method_used == "risk_reward"

    def test_statistical_strategy(self):
        strategy = StatisticalStrategy()
        result = strategy.generate()
        # Default placeholder implementation
        assert result.stop_loss_pct == 0.03
        assert result.take_profit_pct == 0.06
        assert result.method_used == "statistical"

    def test_volatility_strategy(self):
        strategy = VolatilityStrategy()
        result = strategy.generate(
            base_atr_pct=0.01, atr_multiplier_sl=2.0, atr_multiplier_tp=4.0
        )
        assert result.stop_loss_pct == 0.02
        assert result.take_profit_pct == 0.04
        assert result.method_used == "volatility"

    def test_fixed_percentage_strategy(self):
        strategy = FixedPercentageStrategy()
        result = strategy.generate(stop_loss_pct=0.05, take_profit_pct=0.10)
        assert result.stop_loss_pct == 0.05
        assert result.take_profit_pct == 0.10
        assert result.method_used == "fixed_percentage"

    def test_adaptive_strategy_implementation(self):
        """AdaptiveStrategyクラスの直接テスト"""
        # Note: UnifiedTPSLGenerator uses FixedPercentageStrategy for adaptive key currently,
        # but the class AdaptiveStrategy exists.
        strategy = AdaptiveStrategy()

        # 1. Test delegation to Volatility via TPSLGene
        mock_gene_vol = Mock()
        mock_gene_vol.method.name = "VOLATILITY_BASED"
        result = strategy.generate(tpsl_gene=mock_gene_vol, base_atr_pct=0.01)
        assert result.method_used == "volatility"
        assert result.stop_loss_pct == 0.015  # 0.01 * 1.5 default

        # 2. Test delegation to RiskReward
        mock_gene_rr = Mock()
        mock_gene_rr.method.name = "RISK_REWARD_RATIO"
        result = strategy.generate(
            tpsl_gene=mock_gene_rr, stop_loss_pct=0.02, target_ratio=2.0
        )
        assert result.method_used == "risk_reward"
        assert result.take_profit_pct == 0.04

        # 3. Test delegation to Statistical
        mock_gene_stat = Mock()
        mock_gene_stat.method.name = "STATISTICAL"
        result = strategy.generate(tpsl_gene=mock_gene_stat)
        assert result.method_used == "statistical"

        # 4. Fallback
        mock_gene_other = Mock()
        mock_gene_other.method.name = "UNKNOWN"
        result = strategy.generate(tpsl_gene=mock_gene_other, stop_loss_pct=0.01)
        assert result.method_used == "fixed_percentage"
        assert result.stop_loss_pct == 0.01


class TestUnifiedTPSLGenerator:
    """UnifiedTPSLGeneratorのテスト"""

    @pytest.fixture
    def generator(self):
        return UnifiedTPSLGenerator()

    def test_generate_tpsl_methods(self, generator):
        """メソッド指定による生成"""
        # Risk Reward
        res_rr = generator.generate_tpsl("risk_reward", stop_loss_pct=0.01)
        assert res_rr.method_used == "risk_reward"

        # Statistical
        res_stat = generator.generate_tpsl("statistical")
        assert res_stat.method_used == "statistical"

        # Volatility
        res_vol = generator.generate_tpsl("volatility")
        assert res_vol.method_used == "volatility"

        # Fixed
        res_fixed = generator.generate_tpsl("fixed_percentage")
        assert res_fixed.method_used == "fixed_percentage"

        # Unknown
        with pytest.raises(ValueError, match="Unknown TPSL method"):
            generator.generate_tpsl("invalid_method")

    def test_generate_adaptive_tpsl_logic(self, generator):
        """市場条件に基づく自動選択ロジック"""

        # High volatility -> Volatility
        res_vol = generator.generate_adaptive_tpsl({"volatility": "high"})
        assert res_vol.method_used == "volatility"

        # Strong trend -> Risk Reward
        res_rr = generator.generate_adaptive_tpsl({"trend": "strong_up"})
        assert res_rr.method_used == "risk_reward"

        # Historical data -> Statistical
        res_stat = generator.generate_adaptive_tpsl({"historical_data_available": True})
        assert res_stat.method_used == "statistical"

        # Default -> Fixed
        res_def = generator.generate_adaptive_tpsl({})
        assert res_def.method_used == "fixed_percentage"

    def test_adaptive_key_mapping(self, generator):
        """TPSLMethod.ADAPTIVEキーの挙動確認"""
        # Current implementation maps ADAPTIVE to specific strategy (currently FixedPercentageStrategy in _create_adaptive_strategy)
        # Verify what it returns
        res = generator.generate_tpsl("adaptive", stop_loss_pct=0.04)
        # Based on _create_adaptive_strategy returning FixedPercentageStrategy
        assert res.method_used == "fixed_percentage"
        assert res.stop_loss_pct == 0.04


