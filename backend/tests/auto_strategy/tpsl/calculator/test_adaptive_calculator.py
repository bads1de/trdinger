"""
AdaptiveCalculator のテスト (修正版)
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.auto_strategy.tpsl.calculator.adaptive_calculator import (
    AdaptiveCalculator,
)
from app.services.auto_strategy.genes.tpsl_gene import TPSLGene
from app.services.auto_strategy.config.enums import TPSLMethod
from app.services.auto_strategy.genes import TPSLResult


class TestAdaptiveCalculator:
    """AdaptiveCalculator のテスト"""

    @pytest.fixture
    def calculator(self):
        return AdaptiveCalculator()

    def test_select_best_method_volatility(self, calculator):
        """高ボラティリティ時の選択ロジックテスト"""
        market_data = {"volatility": "high"}
        method = calculator._select_best_method(market_data, None)
        assert method == "volatility"

    def test_select_best_method_trend(self, calculator):
        """トレンド相場時の選択ロジックテスト"""
        market_data = {"volatility": "normal", "trend": "strong_up"}
        method = calculator._select_best_method(market_data, None)
        assert method == "risk_reward"

    def test_select_best_method_statistical(self, calculator):
        """十分なデータがある場合の統計方式選択テスト"""
        market_data = {
            "volatility": "normal",
            "trend": "neutral",
            "historical_data_available": True,
            "historical_prices": [100.0] * 101,  # > 100
        }
        method = calculator._select_best_method(market_data, None)
        assert method == "statistical"

    def test_select_best_method_gene_override(self, calculator):
        """Gene指定による強制選択テスト"""
        # 市場環境がAdaptiveな要因を持たない場合、Geneの指定が優先されるか確認
        market_data = {"volatility": "normal", "trend": "neutral"}

        # 確実にマップにある統計方式を使用
        gene_stat = TPSLGene(method=TPSLMethod.STATISTICAL)
        method = calculator._select_best_method(market_data, gene_stat)
        assert method == "statistical"

    def test_select_best_method_default(self, calculator):
        """デフォルトフォールバックテスト"""
        market_data = {}
        method = calculator._select_best_method(market_data, None)
        assert method == "fixed_percentage"

    def test_calculate_delegation(self, calculator):
        """計算委譲のテスト"""
        market_data = {"volatility": "high"}

        # モックが本物のResultオブジェクトを返すようにする
        mock_result = TPSLResult(
            stop_loss_pct=0.05,
            take_profit_pct=0.1,
            method_used="volatility",
            expected_performance={},
            metadata={},
        )

        with patch.object(
            calculator.calculators["volatility"], "calculate"
        ) as mock_calc:
            mock_calc.return_value = mock_result

            result = calculator.calculate(current_price=100.0, market_data=market_data)

            mock_calc.assert_called_once()
            # 委譲後に情報が付与されていること
            assert result.expected_performance["adaptive_selection"] == "volatility"
            assert result.metadata["selected_method"] == "volatility"
            assert result.stop_loss_pct == 0.05

    def test_calculate_fallback(self, calculator):
        """計算エラー時のフォールバックテスト"""
        market_data = {"volatility": "high"}

        # calculate で例外発生
        with patch.object(
            calculator.calculators["volatility"],
            "calculate",
            side_effect=Exception("Error"),
        ):
            result = calculator.calculate(current_price=100.0, market_data=market_data)

            # フォールバック処理が走る
            assert result.stop_loss_pct == 0.03
            assert result.take_profit_pct == 0.06




