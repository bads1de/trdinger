"""
Volatility Calculator Tests
"""

import pytest
from unittest.mock import Mock

from app.services.auto_strategy.tpsl.calculator.volatility_calculator import (
    VolatilityCalculator,
)
from app.services.auto_strategy.models import TPSLGene


class TestVolatilityCalculator:
    """VolatilityCalculatorのテスト"""

    @pytest.fixture
    def calculator(self):
        return VolatilityCalculator()

    def test_calculate_with_gene_and_market_data_atr(self, calculator):
        """Geneと市場データ（ATR直接）を使用して計算"""
        gene = Mock(spec=TPSLGene)
        gene.atr_period = 14
        gene.atr_multiplier_sl = 2.0
        gene.atr_multiplier_tp = 4.0

        market_data = {"atr": 5.0}  # Price=100 -> 5%

        result = calculator.calculate(
            current_price=100.0, tpsl_gene=gene, market_data=market_data
        )

        # Base pct = 5.0 / 100.0 = 0.05
        # SL = 0.05 * 2.0 = 0.10
        # TP = 0.05 * 4.0 = 0.20
        assert result.stop_loss_pct == 0.10
        assert result.take_profit_pct == 0.20
        assert result.expected_performance["atr_value"] == 5.0

    def test_calculate_with_ohlc(self, calculator):
        """OHLCデータからATRを計算"""
        # Simple OHLC data for calculation
        # TR = max(h-l, |h-cp|, |l-cp|)
        # Make constant TR for simplicity
        # Prev close: 100. High: 110. Low: 90. TR = 20.
        ohlc = [
            {"close": 100, "high": 105, "low": 95},  # Initial
        ]
        # Generate 15 candles
        for _ in range(15):
            ohlc.append({"close": 100, "high": 110, "low": 90})

        market_data = {"ohlc_data": ohlc}

        # atr_period = 10
        result = calculator.calculate(
            current_price=100.0,
            tpsl_gene=None,  # Use default kwargs/defaults
            market_data=market_data,
            atr_period=10,
            atr_multiplier_sl=1.0,
            atr_multiplier_tp=2.0,
        )

        # TR should be 20 for last 10 candles. ATR=20.
        # Base pct = 20 / 100 = 0.2
        # SL = 0.2 * 1.0 = 0.2
        # TP = 0.2 * 2.0 = 0.4

        assert result.expected_performance["atr_value"] == 20.0
        assert result.stop_loss_pct == 0.2
        assert result.take_profit_pct == 0.4

    def test_calculate_with_volatility_fallback(self, calculator):
        """ボラティリティ推定（ATRなし）"""
        market_data = {"volatility": 0.05}

        result = calculator.calculate(
            current_price=100.0,
            market_data=market_data,
            atr_multiplier_sl=1.0,
            atr_multiplier_tp=2.0,
        )

        # ATR estimated as price * volatility = 100 * 0.05 = 5.0
        # Base pct = 5 / 100 = 0.05
        assert result.expected_performance["atr_value"] == 5.0
        assert result.stop_loss_pct == 0.05
        assert result.take_profit_pct == 0.10

    def test_calculate_defaults_no_market_data(self, calculator):
        """市場データなしの場合のデフォルト"""
        result = calculator.calculate(current_price=100.0)

        # base_atr_pct defaults to 0.02 if no ATR
        # Default multipliers: SL=1.5, TP=3.0 (from kwargs.get defaults in code)

        assert result.stop_loss_pct == 0.02 * 1.5  # 0.03
        assert result.take_profit_pct == 0.02 * 3.0  # 0.06

    def test_error_handling(self, calculator):
        """エラーハンドリング"""
        calculator._get_atr_value = Mock(side_effect=Exception("Error"))

        result = calculator.calculate(current_price=100.0)

        assert result.expected_performance["type"] == "volatility_fallback"
        assert result.confidence_score == 0.5


