"""
ポジションサイジング計算機のテスト
"""

import pytest
from unittest.mock import Mock, patch

from app.services.auto_strategy.positions.calculators.fixed_quantity_calculator import (
    FixedQuantityCalculator,
)
from app.services.auto_strategy.positions.calculators.fixed_ratio_calculator import (
    FixedRatioCalculator,
)
from app.services.auto_strategy.positions.calculators.volatility_based_calculator import (
    VolatilityBasedCalculator,
)


class TestFixedQuantityCalculator:
    """FixedQuantityCalculatorのテスト"""

    def test_calculate(self):
        gene = Mock()
        gene.fixed_quantity = 0.5
        gene.min_position_size = 0.01
        gene.max_position_size = 100.0

        calculator = FixedQuantityCalculator()

        result = calculator.calculate(
            gene=gene, account_balance=10000, current_price=50000
        )

        assert result["position_size"] == 0.5
        assert result["details"]["method"] == "fixed_quantity"


class TestFixedRatioCalculator:
    """FixedRatioCalculatorのテスト"""

    def test_calculate(self):
        gene = Mock()
        gene.fixed_ratio = 0.1  # 残高の10%
        gene.min_position_size = 0.01
        gene.max_position_size = 100.0

        calculator = FixedRatioCalculator()

        result = calculator.calculate(
            gene=gene, account_balance=10000, current_price=100  # 価格100
        )

        # 10000 * 0.1 / 100 = 10.0
        assert result["position_size"] == 10.0
        assert result["details"]["calculated_amount"] == 1000.0


class TestVolatilityBasedCalculator:
    """VolatilityBasedCalculatorのテスト"""

    @pytest.fixture
    def calculator(self):
        return VolatilityBasedCalculator()

    @pytest.fixture
    def gene(self):
        gene = Mock()
        gene.risk_per_trade = 0.02  # 2% risk
        gene.atr_multiplier = 2.0
        gene.min_position_size = 0.01
        gene.max_position_size = 100.0
        # VaR関連のデフォルト
        gene.var_confidence = 0.95
        gene.var_lookback = 20
        gene.max_var_ratio = 0.0  # 無効
        gene.max_expected_shortfall_ratio = 0.0  # 無効
        return gene

    def test_calculate_basic(self, calculator, gene):
        """基本的なATRベースの計算"""
        # 口座10000, 価格100, ATR=2 (2%)
        # リスク許容額 = 10000 * 0.02 = 200
        # 1株あたりのリスク = price * atr_pct * multiplier = 100 * 0.02 * 2 = 4
        # サイズ = 200 / 4 = 50

        market_data = {"atr": 2.0}

        result = calculator.calculate(
            gene=gene,
            account_balance=10000,
            current_price=100.0,
            market_data=market_data,
        )

        assert result["position_size"] == pytest.approx(50.0)
        assert result["details"]["risk_amount"] == 200.0

    def test_calculate_with_var_constraint(self, calculator, gene):
        """VaR制約の適用テスト"""
        # VaR制限を有効化
        gene.max_var_ratio = 0.01  # 最大1%のVaR

        market_data = {"atr": 2.0, "returns": [0.01, -0.01, 0.02]}

        with patch(
            "app.services.auto_strategy.positions.calculators.volatility_based_calculator.calculate_historical_var",
            return_value=0.1,
        ) as mock_var:
            # var_ratio = 0.1 (10%)
            # 本来のポジションサイズでのVaR損失 = 50株 * 100円 * 0.1 = 500円
            # 許容VaR損失 = 10000 * 0.01 = 100円
            # 制限されるはず

            result = calculator.calculate(
                gene=gene,
                account_balance=10000,
                current_price=100.0,
                market_data=market_data,
            )

            assert result["details"]["risk_controls"]["var_adjusted"] is True
            # 制限後のサイズ = 100円 / 0.1 = 1000円 (ポジション価値) => 10株
            assert result["position_size"] == pytest.approx(10.0)




