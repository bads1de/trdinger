"""
Risk Reward Calculator Tests
"""

import pytest
from unittest.mock import Mock

from backend.app.services.auto_strategy.tpsl.calculator.risk_reward_calculator import (
    RiskRewardCalculator,
)
from backend.app.services.auto_strategy.models.strategy_models import TPSLGene


class TestRiskRewardCalculator:
    """RiskRewardCalculatorのテスト"""

    @pytest.fixture
    def calculator(self):
        return RiskRewardCalculator()

    def test_calculate_with_gene(self, calculator):
        """Geneを使用して計算"""
        gene = Mock(spec=TPSLGene)
        gene.base_stop_loss = 0.02
        gene.risk_reward_ratio = 3.0

        result = calculator.calculate(
            current_price=100.0, tpsl_gene=gene, position_direction=1.0
        )

        assert result.stop_loss_pct == 0.02
        assert result.take_profit_pct == 0.06  # 0.02 * 3.0
        assert result.expected_performance["risk_reward_ratio"] == 3.0

    def test_calculate_with_gene_fallback_attribute(self, calculator):
        """Geneのstop_loss_pctを使用（base_stop_lossがNoneの場合）"""
        gene = Mock(spec=TPSLGene)
        gene.base_stop_loss = None
        gene.stop_loss_pct = 0.025
        gene.risk_reward_ratio = 2.0

        result = calculator.calculate(current_price=100.0, tpsl_gene=gene)

        assert result.stop_loss_pct == 0.025
        assert result.take_profit_pct == 0.05

    def test_calculate_with_kwargs(self, calculator):
        """kwargsを使用して計算"""
        result = calculator.calculate(
            current_price=100.0, tpsl_gene=None, base_stop_loss=0.04, target_ratio=2.5
        )

        assert result.stop_loss_pct == 0.04
        assert result.take_profit_pct == 0.1  # 0.04 * 2.5

    def test_calculate_defaults(self, calculator):
        """デフォルト値"""
        result = calculator.calculate(current_price=100.0, tpsl_gene=None)

        # Defaults: sl=0.03, ratio=2.0
        assert result.stop_loss_pct == 0.03
        assert result.take_profit_pct == 0.06
