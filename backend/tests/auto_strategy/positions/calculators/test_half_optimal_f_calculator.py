"""
HalfOptimalFCalculator のテスト
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.auto_strategy.positions.calculators.half_optimal_f_calculator import (
    HalfOptimalFCalculator,
)
from app.services.auto_strategy.genes.position_sizing import PositionSizingGene
from app.services.auto_strategy.config.enums import PositionSizingMethod


class TestHalfOptimalFCalculator:
    """HalfOptimalFCalculator のテストクラス"""

    @pytest.fixture
    def calculator(self):
        return HalfOptimalFCalculator()

    @pytest.fixture
    def sample_gene(self):
        return PositionSizingGene(
            method=PositionSizingMethod.HALF_OPTIMAL_F,
            optimal_f_multiplier=0.5,
            fixed_ratio=0.1,
            lookback_period=10,
            enabled=True,
        )

    def test_calculate_with_sufficient_history(self, calculator, sample_gene):
        """十分な取引履歴がある場合の計算テスト"""
        account_balance = 100000.0
        current_price = 100.0

        trade_history = []
        for _ in range(5):
            trade_history.append({"pnl": 200})
        for _ in range(5):
            trade_history.append({"pnl": -100})

        result = calculator.calculate(
            gene=sample_gene,
            account_balance=account_balance,
            current_price=current_price,
            trade_history=trade_history,
        )

        details = result["details"]
        assert details["method"] == "half_optimal_f"
        assert details["win_rate"] == 0.5
        assert details["avg_win"] == 200
        assert details["avg_loss"] == 100
        assert pytest.approx(details["optimal_f"]) == 0.25
        assert pytest.approx(details["half_optimal_f"]) == 0.125

        expected_position_size = (account_balance * 0.125) / current_price
        assert pytest.approx(result["position_size"]) == expected_position_size

    def test_calculate_insufficient_history(self, calculator, sample_gene):
        """取引履歴が不足している場合（簡易計算フォールバック）のテスト"""
        account_balance = 100000.0
        current_price = 100.0
        trade_history = [{"pnl": 100}] * 5  # 10件未満

        result = calculator.calculate(
            gene=sample_gene,
            account_balance=account_balance,
            current_price=current_price,
            trade_history=trade_history,
        )

        details = result["details"]
        assert "fallback_reason" in details
        assert details["fallback_reason"] == "insufficient_trade_history_simplified"

    def test_calculate_no_history(self, calculator, sample_gene):
        """取引履歴がない場合（簡易計算）のテスト"""
        result = calculator.calculate(
            gene=sample_gene,
            account_balance=100000.0,
            current_price=100.0,
            trade_history=[],
        )

        details = result["details"]
        assert details["fallback_reason"] == "insufficient_trade_history_simplified"

    def test_calculate_invalid_data(self, calculator, sample_gene):
        """無効なデータ（全て負けなど）の場合のフォールバックテスト"""
        account_balance = 100000.0
        current_price = 100.0
        trade_history = [{"pnl": -100}] * 10

        result = calculator.calculate(
            gene=sample_gene,
            account_balance=account_balance,
            current_price=current_price,
            trade_history=trade_history,
        )

        details = result["details"]
        # 全て負けの場合 wins が空になるため no_valid_trades になるのが正しい挙動
        assert details.get("fallback_reason") == "no_valid_trades"
        assert details.get("fallback_ratio") == 0.1

    def test_simplified_calculation_failure(self, calculator, sample_gene):
        """簡易計算すら失敗した場合の固定比率フォールバック"""
        with patch.object(
            calculator,
            "_safe_calculate_with_price_check",
            side_effect=Exception("Calc Error"),
        ):
            # 履歴不足
            trade_history = [{"pnl": 100}] * 5

            result = calculator._calculate_simplified_optimal_f(
                gene=sample_gene,
                account_balance=10000.0,
                current_price=100.0,
                trade_history=trade_history,
                warnings=[],
                details={},
            )

            assert (
                result["details"]["fallback_reason"] == "simplified_calculation_failed"
            )




