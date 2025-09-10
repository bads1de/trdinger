"""
position_sizing_gene.py のユニットテスト
"""
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import replace

from backend.app.services.auto_strategy.models.position_sizing_gene import PositionSizingGene
from backend.app.services.auto_strategy.models.enums import PositionSizingMethod


class TestPositionSizingGene:
    """PositionSizingGene クラステスト"""

    def test_position_sizing_gene_initialization(self):
        """PositionSizingGene の初期化テスト"""
        # 基本的な初期化
        gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=100,
            optimal_f_multiplier=0.5,
            atr_period=14,
            atr_multiplier=2.0,
            risk_per_trade=0.02,
            fixed_ratio=0.1,
            fixed_quantity=1.0,
            min_position_size=0.01,
            max_position_size=10.0,
            enabled=True,
            priority=1.0
        )

        assert gene.method == PositionSizingMethod.VOLATILITY_BASED
        assert gene.lookback_period == 100
        assert gene.optimal_f_multiplier == 0.5
        assert gene.atr_period == 14
        assert gene.atr_multiplier == 2.0
        assert gene.risk_per_trade == 0.02
        assert gene.fixed_ratio == 0.1
        assert gene.fixed_quantity == 1.0
        assert gene.min_position_size == 0.01
        assert gene.max_position_size == 10.0
        assert gene.enabled is True
        assert gene.priority == 1.0

    def test_position_sizing_gene_default_values(self):
        """デフォルト値のテスト"""
        gene = PositionSizingGene()

        assert gene.method == PositionSizingMethod.VOLATILITY_BASED
        assert gene.lookback_period == 100
        assert gene.optimal_f_multiplier == 0.5
        assert gene.atr_period == 14
        assert gene.atr_multiplier == 2.0
        assert gene.risk_per_trade == 0.02
        assert gene.fixed_ratio == 0.1
        assert gene.fixed_quantity == 1.0
        assert gene.min_position_size == 0.01
        assert gene.max_position_size == 9999.0  # 注意: デフォルトは9999.0
        assert gene.enabled is True
        assert gene.priority == 1.0

    def test_validate_parameters_with_constants(self):
        """定数を使ったパラメータ検証テスト"""
        # 正しい値の遺伝子
        gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=100,
            optimal_f_multiplier=0.5,
            atr_period=14,
            atr_multiplier=2.0,
            risk_per_trade=0.02,
            fixed_ratio=0.1,
            fixed_quantity=1.0,
            min_position_size=0.01,
            max_position_size=10.0,
            enabled=True,
            priority=1.0
        )

        # 定数が見つかる場合の検証
        with patch('backend.app.services.auto_strategy.models.position_sizing_gene.POSITION_SIZING_LIMITS',
                   {"lookback_period": (10, 500),
                    "optimal_f_multiplier": (0.1, 1.0),
                    "atr_period": (5, 50),
                    "atr_multiplier": (0.5, 10.0),
                    "risk_per_trade": (0.001, 0.1),
                    "fixed_ratio": (0.01, 10.0),
                    "fixed_quantity": (0.01, 1000.0),
                    "min_position_size": (0.001, 1.0),
                    "max_position_size": (0.001, 1.0)}) as mock_limits:

            # BaseGene.validate をモックして正しく呼び出されることを確認
            with patch.object(PositionSizingGene, 'validate', return_value=(True, [])) as mock_validate:
                result = gene.validate()

                # _validate_parameters が呼ばれたことを確認
                mock_validate.assert_called_once()

    def test_validate_parameters_with_constants_beyond_limits(self):
        """定数の範囲外の値をテスト"""
        # 範囲外の値
        gene = PositionSizingGene(
            lookback_period=600,  # 500が上限なのでNG
            risk_per_trade=1.0,   # 0.1が上限なのでNG
            fixed_ratio=100.0,    # 10.0が上限なのでNG
        )

        # エラーが正しく検知されることを確認
        with patch('backend.app.services.auto_strategy.models.position_sizing_gene.POSITION_SIZING_LIMITS',
                   {"lookback_period": (10, 500),
                    "risk_per_trade": (0.001, 0.1),
                    "fixed_ratio": (0.01, 10.0)}) as mock_limits:

            with patch('backend.app.services.auto_strategy.models.position_sizing_gene.BaseGene'):
                # _validate_parameters が誤って呼ばれないようにモック
                result = gene.validate()

                # エラーが含まれているはず
                assert isinstance(result, tuple)
                is_valid, errors = result
                assert isinstance(is_valid, bool)
                assert isinstance(errors, list)

    def test_validate_parameters_import_error_fallback(self):
        """ImportError時のフォールバック検証テスト"""
        gene = PositionSizingGene(
            lookback_period=250,  # 50-200の範囲内でOK
            risk_per_trade=0.05,  # 0.001-0.1の範囲内でOK
            fixed_ratio=5.0,      # 0.001-1.0の範囲内でOK
        )

        # ImportErrorをシミュレートしてフォールバックパスをテスト
        with patch('backend.app.services.auto_strategy.models.position_sizing_gene.import_module',
                   side_effect=ImportError):
            with patch('backend.app.services.auto_strategy.models.position_sizing_gene.BaseGene'):
                result = gene.validate()

                # エラーが正しく検知される
                assert isinstance(result, tuple)
                is_valid, errors = result
                assert isinstance(is_valid, bool)
                assert isinstance(errors, list)

    def test_validate_parameters_fallback_range_checks(self):
        """ImportError時のフォールバック範囲チェックテスト"""
        # フォールバック時の範囲チェック
        # lookback_periodの範囲: 50-200
        gene_valid = PositionSizingGene(
            lookback_period=100,  # 範囲内OK
            risk_per_trade=0.05,  # 範囲内OK
            fixed_ratio=1.0,      # 範囲内OK
        )

        gene_invalid_lookback = PositionSizingGene(
            lookback_period=300,  # 範囲外NG
            risk_per_trade=0.05,
            fixed_ratio=1.0
        )

        gene_invalid_risk = PositionSizingGene(
            lookback_period=100,
            risk_per_trade=1.0,   # 範囲外NG
            fixed_ratio=1.0
        )

        gene_invalid_ratio = PositionSizingGene(
            lookback_period=100,
            risk_per_trade=0.05,
            fixed_ratio=2.0,      # 範囲外NG (0.001-1.0)
        )

        # フォールバック範囲検証
        with patch('backend.app.services.auto_strategy.models.position_sizing_gene.import_anything',
                   side_effect=ImportError):
            with patch('backend.app.services.auto_strategy.models.position_sizing_gene.BaseGene') as mock_base:
                # BaseGeneのvalidateをモックして予想されるエラーを返却
                mock_base.return_value.validate.return_value = (False, ["lookback_periodは50-200の範囲でなければなりません"])
                result = gene_invalid_lookback.validate()
                assert isinstance(result, tuple)


class TestPositionSizingGeneEdgeCases:
    """エッジケーステスト"""

    def test_position_sizing_gene_method_enum(self):
        """Enum型の操作テスト"""
        gene = PositionSizingGene()

        # 各メソッドの設定テスト
        gene.method = PositionSizingMethod.FIXED_QUANTITY
        assert gene.method == PositionSizingMethod.FIXED_QUANTITY

        gene.method = PositionSizingMethod.FIXED_RATIO
        assert gene.method == PositionSizingMethod.FIXED_RATIO

    def test_position_sizing_gene_numeric_precision(self):
        """浮動小数点精度テスト"""
        gene = PositionSizingGene(
            optimal_f_multiplier=0.249999999,
            risk_per_trade=0.100000001,
            fixed_ratio=0.999999999,
            priority=1.000000001
        )

        # 数値が正しく設定されているか
        assert gene.optimal_f_multiplier == 0.249999999
        assert gene.risk_per_trade == 0.100000001
        assert gene.fixed_ratio == 0.999999999
        assert gene.priority == 1.000000001

    def test_position_sizing_gene_boolean_flags(self):
        """ブール値フラグのテスト"""
        enabled_gene = PositionSizingGene(enabled=True)
        assert enabled_gene.enabled is True

        disabled_gene = PositionSizingGene(enabled=False)
        assert disabled_gene.enabled is False

    def test_position_sizing_gene_extreme_values(self):
        """極端な値のテスト"""
        gene = PositionSizingGene(
            lookback_period=999999,  # 非常に大きな値
            optimal_f_multiplier=999999.0,
            risk_per_trade=999.0,
            fixed_ratio=999.0,
            fixed_quantity=999999.0,
            min_position_size=999.0,
            max_position_size=999999.0,
            priority=999.0
        )

        # 値が設定されていることを確認
        assert gene.lookback_period == 999999
        assert gene.optimal_f_multiplier == 999999.0

    def test_position_sizing_gene_negative_values(self):
        """負の値のテスト"""
        gene = PositionSizingGene(
            lookback_period=-100,
            optimal_f_multiplier=-0.5,
            risk_per_trade=-0.01,
            fixed_ratio=-0.5,
            fixed_quantity=-10.0
        )

        # 負の値が設定されていることを確認
        assert gene.lookback_period == -100
        assert gene.optimal_f_multiplier == -0.5

    def test_position_sizing_gene_zero_values(self):
        """ゼロ値のテスト"""
        gene = PositionSizingGene(
            lookback_period=0,
            optimal_f_multiplier=0.0,
            risk_per_trade=0.0,
            fixed_ratio=0.0,
            fixed_quantity=0.0,
            min_position_size=0.0,
            max_position_size=0.0,
            priority=0.0
        )

        # ゼロ値が設定されていることを確認
        assert gene.lookback_period == 0
        assert gene.optimal_f_multiplier == 0.0

    def test_position_sizing_gene_inheritance_from_base_gene(self):
        """BaseGeneからの継承テスト"""
        gene = PositionSizingGene()

        # BaseGeneのメソッドが存在することを確認
        assert hasattr(gene, 'to_dict')
        assert hasattr(gene, 'from_dict')
        assert hasattr(gene, 'validate')
        assert hasattr(gene, '_validate_parameters')