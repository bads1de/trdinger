"""
Fixed Percentage Calculator Tests
"""

import pytest
from unittest.mock import Mock

from app.services.auto_strategy.tpsl.calculator.fixed_percentage_calculator import (
    FixedPercentageCalculator,
)
from app.services.auto_strategy.genes import TPSLGene


class TestFixedPercentageCalculator:
    """FixedPercentageCalculatorのテスト"""

    @pytest.fixture
    def calculator(self):
        return FixedPercentageCalculator()

    def test_calculate_with_gene(self, calculator):
        """Geneを使用して計算"""
        gene = Mock(spec=TPSLGene)
        gene.stop_loss_pct = 0.05
        gene.take_profit_pct = 0.10

        result = calculator.calculate(
            current_price=100.0, tpsl_gene=gene, position_direction=1.0
        )

        assert result.stop_loss_pct == 0.05
        assert result.take_profit_pct == 0.10
        assert result.expected_performance["tp_price"] == pytest.approx(110.0)
        assert result.expected_performance["sl_price"] == pytest.approx(95.0)

        # Short
        result_short = calculator.calculate(
            current_price=100.0, tpsl_gene=gene, position_direction=-1.0
        )
        assert result_short.expected_performance["tp_price"] == pytest.approx(90.0)
        assert result_short.expected_performance["sl_price"] == pytest.approx(105.0)

    def test_calculate_with_kwargs(self, calculator):
        """kwargsを使用して計算（Geneなし）"""
        result = calculator.calculate(
            current_price=100.0,
            tpsl_gene=None,
            position_direction=1.0,
            stop_loss_pct=0.04,
            take_profit_pct=0.08,
        )

        assert result.stop_loss_pct == 0.04
        assert result.take_profit_pct == 0.08

    def test_calculate_default_fallback(self, calculator):
        """デフォルトフォールバック"""
        # kwargsもなし
        result = calculator.calculate(
            current_price=100.0, tpsl_gene=None, position_direction=1.0
        )
        # Default in code: sl=0.03, tp=0.06
        assert result.stop_loss_pct == 0.03
        assert result.take_profit_pct == 0.06




