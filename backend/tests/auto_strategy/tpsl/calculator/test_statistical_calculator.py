"""
StatisticalCalculator のテスト (修正版)
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.auto_strategy.tpsl.calculator.statistical_calculator import (
    StatisticalCalculator,
)
from app.services.auto_strategy.models.tpsl_gene import TPSLGene
from app.services.auto_strategy.models import TPSLResult


class TestStatisticalCalculator:
    """StatisticalCalculator のテスト"""

    @pytest.fixture
    def calculator(self):
        return StatisticalCalculator()

    def test_calculate_with_sufficient_history(self, calculator):
        """十分な履歴データがある場合の計算テスト"""
        current_price = 100.0
        historical_prices = [100.0 * (1 + 0.01 * (i % 2)) for i in range(200)]
        market_data = {"historical_prices": historical_prices}

        gene = TPSLGene(lookback_period=100, confidence_threshold=0.95)

        result = calculator.calculate(
            current_price=current_price, tpsl_gene=gene, market_data=market_data
        )

        assert isinstance(result, TPSLResult)
        assert result.method_used == "statistical"
        assert result.stop_loss_pct > 0
        assert result.take_profit_pct > 0
        assert result.stop_loss_pct != 0.03
        assert result.take_profit_pct != 0.06

    def test_calculate_insufficient_history(self, calculator):
        """履歴データが不足している場合のテスト"""
        current_price = 100.0
        market_data = {"historical_prices": [100.0] * 10}
        gene = TPSLGene(lookback_period=100)

        result = calculator.calculate(
            current_price=current_price, tpsl_gene=gene, market_data=market_data
        )

        assert result.stop_loss_pct == 0.03
        assert result.take_profit_pct == 0.06

    def test_calculate_no_history(self, calculator):
        """履歴データがない場合のテスト"""
        current_price = 100.0
        market_data = {}
        result = calculator.calculate(
            current_price=current_price, market_data=market_data
        )
        assert result.stop_loss_pct == 0.03
        assert result.take_profit_pct == 0.06

    def test_calculate_error_handling(self, calculator):
        """計算エラー時のフォールバックテスト"""
        # calculate内で例外を起こさせる
        with patch.object(
            calculator,
            "_calculate_statistical_levels",
            side_effect=Exception("Critical Error"),
        ):
            result = calculator.calculate(
                current_price=100.0, market_data={"historical_prices": [100.0] * 200}
            )
            # method_usedはクラス初期化時のものが使われる仕様を確認済み
            assert result.method_used == "statistical"
            # その代わり、expected_performance["type"] が fallback を示す
            assert result.expected_performance["type"] == "statistical_fallback"
            assert result.stop_loss_pct == 0.03

    def test_min_max_constraints(self, calculator):
        """計算結果の最小・最大値制約のテスト"""
        current_price = 100.0

        # Max constraints
        historical_prices = [100.0 * (1 + 1.0 * (i % 2)) for i in range(200)]
        market_data = {"historical_prices": historical_prices}

        result = calculator.calculate(
            current_price=current_price, market_data=market_data, lookback_period=100
        )

        assert result.stop_loss_pct == 0.1
        assert result.take_profit_pct == 0.2

        # Min constraints
        historical_prices = [100.0 for i in range(200)]
        market_data = {"historical_prices": historical_prices}

        result = calculator.calculate(
            current_price=current_price, market_data=market_data, lookback_period=100
        )

        assert result.stop_loss_pct == 0.01
        assert result.take_profit_pct == 0.02


