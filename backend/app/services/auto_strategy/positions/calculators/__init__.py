"""
ポジションサイジング計算機パッケージ

各計算手法の計算クラスを提供します。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_calculator import BaseCalculator
    from .calculator_factory import CalculatorFactory
    from .fixed_quantity_calculator import FixedQuantityCalculator
    from .fixed_ratio_calculator import FixedRatioCalculator
    from .half_optimal_f_calculator import HalfOptimalFCalculator
    from .volatility_based_calculator import VolatilityBasedCalculator

_ATTRIBUTE_EXPORTS = {
    "BaseCalculator": ".base_calculator",
    "CalculatorFactory": ".calculator_factory",
    "FixedQuantityCalculator": ".fixed_quantity_calculator",
    "FixedRatioCalculator": ".fixed_ratio_calculator",
    "HalfOptimalFCalculator": ".half_optimal_f_calculator",
    "VolatilityBasedCalculator": ".volatility_based_calculator",
}

__all__ = [
    "BaseCalculator",
    "CalculatorFactory",
    "FixedQuantityCalculator",
    "FixedRatioCalculator",
    "HalfOptimalFCalculator",
    "VolatilityBasedCalculator",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS)
