"""
CalculatorFactoryのテスト

ポジションサイジング計算機ファクトリのテスト
"""

from unittest.mock import Mock


class TestCalculatorFactoryCreateCalculator:
    """create_calculatorのテスト"""

    def test_creates_half_optimal_f_calculator(self):
        """HalfOptimalFCalculatorを作成"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )
        from app.services.auto_strategy.positions.calculators.half_optimal_f_calculator import (
            HalfOptimalFCalculator,
        )

        calculator = CalculatorFactory.create_calculator("half_optimal_f")
        assert isinstance(calculator, HalfOptimalFCalculator)

    def test_creates_volatility_based_calculator(self):
        """VolatilityBasedCalculatorを作成"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )
        from app.services.auto_strategy.positions.calculators.volatility_based_calculator import (
            VolatilityBasedCalculator,
        )

        calculator = CalculatorFactory.create_calculator("volatility_based")
        assert isinstance(calculator, VolatilityBasedCalculator)

    def test_creates_fixed_ratio_calculator(self):
        """FixedRatioCalculatorを作成"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )
        from app.services.auto_strategy.positions.calculators.fixed_ratio_calculator import (
            FixedRatioCalculator,
        )

        calculator = CalculatorFactory.create_calculator("fixed_ratio")
        assert isinstance(calculator, FixedRatioCalculator)

    def test_creates_fixed_quantity_calculator(self):
        """FixedQuantityCalculatorを作成"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )
        from app.services.auto_strategy.positions.calculators.fixed_quantity_calculator import (
            FixedQuantityCalculator,
        )

        calculator = CalculatorFactory.create_calculator("fixed_quantity")
        assert isinstance(calculator, FixedQuantityCalculator)

    def test_returns_fixed_ratio_for_unknown_method(self):
        """不明なメソッドの場合はFixedRatioCalculatorを返す"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )
        from app.services.auto_strategy.positions.calculators.fixed_ratio_calculator import (
            FixedRatioCalculator,
        )

        calculator = CalculatorFactory.create_calculator("unknown_method")
        assert isinstance(calculator, FixedRatioCalculator)

    def test_handles_enum_method(self):
        """enumメソッドを処理"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )
        from app.services.auto_strategy.positions.calculators.half_optimal_f_calculator import (
            HalfOptimalFCalculator,
        )

        # enum風オブジェクトをモック
        mock_enum = Mock()
        mock_enum.value = "half_optimal_f"

        calculator = CalculatorFactory.create_calculator(mock_enum)
        assert isinstance(calculator, HalfOptimalFCalculator)

    def test_all_calculators_inherit_from_base(self):
        """全ての計算機がBaseCalculatorを継承"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )
        from app.services.auto_strategy.positions.calculators.base_calculator import (
            BaseCalculator,
        )

        methods = [
            "half_optimal_f",
            "volatility_based",
            "fixed_ratio",
            "fixed_quantity",
        ]

        for method in methods:
            calculator = CalculatorFactory.create_calculator(method)
            assert isinstance(
                calculator, BaseCalculator
            ), f"{method} should inherit from BaseCalculator"


class TestCalculatorFactoryGetAvailableMethods:
    """get_available_methodsのテスト"""

    def test_returns_all_available_methods(self):
        """全ての利用可能なメソッドを返す"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )

        methods = CalculatorFactory.get_available_methods()

        assert "half_optimal_f" in methods
        assert "volatility_based" in methods
        assert "fixed_ratio" in methods
        assert "fixed_quantity" in methods

    def test_returns_japanese_descriptions(self):
        """日本語の説明を返す"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )

        methods = CalculatorFactory.get_available_methods()

        assert methods["half_optimal_f"] == "ハーフオプティマルF"
        assert methods["volatility_based"] == "ボラティリティベース"
        assert methods["fixed_ratio"] == "固定比率"
        assert methods["fixed_quantity"] == "固定枚数"

    def test_returns_dict(self):
        """辞書型を返す"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )

        methods = CalculatorFactory.get_available_methods()
        assert isinstance(methods, dict)

    def test_method_count(self):
        """メソッド数を確認"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )

        methods = CalculatorFactory.get_available_methods()
        assert len(methods) == 4


class TestCalculatorFactoryIntegration:
    """統合テスト"""

    def test_created_calculators_can_calculate(self):
        """作成された計算機が計算を実行できる"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )

        methods = CalculatorFactory.get_available_methods()

        for method_key in methods.keys():
            calculator = CalculatorFactory.create_calculator(method_key)
            # calculateメソッドが存在することを確認（BaseCalculatorの抽象メソッド）
            assert hasattr(
                calculator, "calculate"
            ), f"{method_key} should have calculate method"

    def test_create_calculator_is_static(self):
        """create_calculatorが静的メソッド"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )

        # インスタンスなしで呼び出せることを確認
        calculator = CalculatorFactory.create_calculator("fixed_ratio")
        assert calculator is not None

    def test_get_available_methods_is_static(self):
        """get_available_methodsが静的メソッド"""
        from app.services.auto_strategy.positions.calculators.calculator_factory import (
            CalculatorFactory,
        )

        # インスタンスなしで呼び出せることを確認
        methods = CalculatorFactory.get_available_methods()
        assert methods is not None


