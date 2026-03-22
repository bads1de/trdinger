import pytest
from unittest.mock import MagicMock
from app.services.auto_strategy.tpsl.calculator.base_calculator import BaseTPSLCalculator
from app.services.auto_strategy.genes.tpsl import TPSLResult

class MockTPSLCalculator(BaseTPSLCalculator):
    def __init__(self):
        super().__init__("mock_tpsl")
        self.should_fail = False

    def _do_calculate(self, current_price, tpsl_gene, market_data, position_direction, **kwargs):
        if self.should_fail:
            raise Exception("Calculation Failed")
        # sl_pct, tp_pct, confidence, metrics
        return 0.01, 0.02, 0.8, {"test": True}

class TestBaseTPSLCalculator:
    @pytest.fixture
    def calculator(self):
        return MockTPSLCalculator()

    def test_calculate_success(self, calculator):
        result = calculator.calculate(50000.0)
        assert isinstance(result, TPSLResult)
        assert result.stop_loss_pct == 0.01
        assert result.take_profit_pct == 0.02
        assert result.method_used == "mock_tpsl"

    def test_calculate_failure_fallback(self, calculator):
        calculator.should_fail = True
        result = calculator.calculate(50000.0)
        # フォールバック値が返ること (BaseTPSLCalculator._create_fallback_result)
        assert result.stop_loss_pct == 0.03
        assert result.take_profit_pct == 0.06

    def test_make_prices_long(self, calculator):
        # Long: price=100, SL=1%, TP=2%
        # SL price = 100 * (1 - 0.01) = 99
        # TP price = 100 * (1 + 0.02) = 102
        sl, tp = calculator._make_prices(100.0, 0.01, 0.02, 1.0)
        assert sl == pytest.approx(99.0)
        assert tp == pytest.approx(102.0)

    def test_make_prices_short(self, calculator):
        # Short: price=100, SL=1%, TP=2%
        # SL price = 100 * (1 + 0.01) = 101
        # TP price = 100 * (1 - 0.02) = 98
        sl, tp = calculator._make_prices(100.0, 0.01, 0.02, -1.0)
        assert sl == pytest.approx(101.0)
        assert tp == pytest.approx(98.0)

    def test_make_prices_zero(self, calculator):
        sl, tp = calculator._make_prices(100.0, 0.0, 0.0, 1.0)
        assert sl == 100.0
        assert tp == 100.0
        
        sl, tp = calculator._make_prices(100.0, None, None, 1.0)
        assert sl is None
        assert tp is None
