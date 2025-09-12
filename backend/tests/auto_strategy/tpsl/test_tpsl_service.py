"""
TPSL Serviceのテスト（バグ検出）
"""
import pytest
from unittest.mock import Mock

from app.services.auto_strategy.models.enums import TPSLMethod
from app.services.auto_strategy.models.tpsl_gene import TPSLGene
from app.services.auto_strategy.tpsl.tpsl_service import TPSLService


class TestTPSLService:
    """TPSL Serviceのテスト"""

    def test_calculate_tpsl_prices_with_gene_method_comparison_bug(self):
        """TPSLServiceのmethod比較バグ検出テスト（Enum比較のバグ）"""
        service = TPSLService()

        # current_price = 100.0
        # TPSLGene作成 - methodはEnumなので直接比較できるはず
        gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            enabled=True
        )

        # TPSLServiceの_calculate_from_geneでは
        # if tpsl_gene.method == TPSLMethod.FIXED_PERCENTAGE:
        # こういう比較をしている（これは正しい）

        sl_price, tp_price = service._calculate_from_gene(
            current_price=100.0,
            tpsl_gene=gene,
            market_data=None,
            position_direction=1.0
        )

        # FIXED_PERCENTAGEの場合、_calculate_fixed_percentageが呼ばれるはず
        # stop_loss_pct=0.05なので、ロングの場合
        expected_sl = 100.0 * (1 - 0.05)
        expected_tp = 100.0 * (1 + 0.10)

        # Expected: sl_price=95.0, tp_price=110.0
        # Debug: print actual values
        assert sl_price == expected_sl, f"SL mismatch: {sl_price} != {expected_sl}"
        assert tp_price == expected_tp, f"TP mismatch: {tp_price} != {expected_tp}"

    def test_calculate_volatility_based_missing_attributes_bug(self):
        """ボラティリティベース計算のパラメータ不足バグ検出テスト"""
        service = TPSLService()

        # risk_reward_ratioなどの属性がないgene
        gene = TPSLGene(
            method=TPSLMethod.VOLATILITY_BASED,
            enabled=True
            # risk_reward_ratioがないので、tpsl_gene.base_stop_lossがundefined
        )

        # _calculate_volatility_basedはtpsl_gene.base_stop_lossを参照しようとする
        # がTPSLGeneのデフォルトは0.03なので動作するはず
        sl_price, tp_price = service._calculate_from_gene(
            current_price=100.0,
            tpsl_gene=gene,
            market_data=None,
            position_direction=1.0
        )

        # VolatilityStrategyが呼ばれるが、デフォルト値で動作
        assert sl_price is not None
        assert tp_price is not None

    def test_calculate_statistical_bug(self):
        """統計的計算のバグ検出テスト"""
        service = TPSLService()

        gene = TPSLGene(
            method=TPSLMethod.STATISTICAL,
            enabled=True
        )

        # _calculate_statisticalはmarket_conditionsを渡しているが
        # UnifiedGeneratorのstatistical_methodは特別なパラメータを期待
        sl_price, tp_price = service._calculate_from_gene(
            current_price=100.0,
            tpsl_gene=gene,
            market_data=None,
            position_direction=1.0
        )

        # エラーでNone, Noneが返される可能性がある
        # 現状では必ずデフォルト値が返される
        assert sl_price is not None
        assert tp_price is not None

    def test_make_prices_with_none_values(self):
        """make_pricesのNone値処理バグ検出テスト"""
        service = TPSLService()

        # stop_loss_pct=None, take_profit_pct=None
        sl_price, tp_price = service.fixed_percentage_calculator._make_prices(
            current_price=100.0,
            stop_loss_pct=None,
            take_profit_pct=None,
            position_direction=1.0
        )

        # None, Noneが返されるはず
        assert sl_price is None
        assert tp_price is None

    def test_make_prices_with_zero_values(self):
        """make_pricesのゼロ値処理テスト"""
        service = TPSLService()

        # stop_loss_pct=0, take_profit_pct=0（現在の価格で指値）
        sl_price, tp_price = service.fixed_percentage_calculator._make_prices(
            current_price=100.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            position_direction=1.0
        )

        # 0は現在の価格を使用
        assert sl_price == 100.0
        assert tp_price == 100.0

    def test_make_prices_short_position(self):
        """make_pricesのショートポジションバグ検出テスト"""
        service = TPSLService()

        # ショートポジション (-1.0)
        sl_price, tp_price = service.fixed_percentage_calculator._make_prices(
            current_price=100.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            position_direction=-1.0
        )

        # ショートの場合、SLは上昇方向、TPは下降方向
        assert sl_price == 100.0 * (1 + 0.05) == 105.0  # SL上昇
        assert tp_price == 100.0 * (1 - 0.10) == 90.0   # TP下降

    def test_validate_price_negative_value_bug(self):
        """validate_priceの負の値処理バグ検出テスト"""
        service = TPSLService()

        # 負の価格
        is_valid = service._validate_price(-10.0)
        assert is_valid == False

        # ゼロ価格
        is_valid = service._validate_price(0.0)
        assert is_valid == False

        # 正の価格
        is_valid = service._validate_price(100.0)
        assert is_valid == True

    def test_validate_percentage_range_bug(self):
        """validate_percentageの範囲チェックバグ検出テスト"""
        service = TPSLService()

        # 有効範囲内の値
        is_valid = service._validate_percentage(0.03, "SL")
        assert is_valid == True

        # 範囲外の上限超過
        is_valid = service._validate_percentage(0.5, "SL")
        assert is_valid == False

        # 範囲外の下限未満
        is_valid = service._validate_percentage(-0.01, "SL")
        assert is_valid == False

    def test_generate_adaptive_tpsl_service_integration(self):
        """TPSLServiceの適応的生成統合テスト"""
        service = TPSLService()

        # 市場条件
        market_conditions = {"volatility": "normal"}

        # AdaptiveCalculatorを直接使用してテスト
        from app.services.auto_strategy.tpsl.calculator.adaptive_calculator import AdaptiveCalculator
        calculator = AdaptiveCalculator()

        # calculateメソッドでTPSLを生成
        result = calculator.calculate(
            current_price=100.0,
            market_data=market_conditions
        )

        assert isinstance(result, object)  # TPSLResult
        assert hasattr(result, 'stop_loss_pct')
        assert hasattr(result, 'take_profit_pct')
        assert hasattr(result, 'method_used')
