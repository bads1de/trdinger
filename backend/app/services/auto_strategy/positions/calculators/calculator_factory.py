"""
計算機ファクトリ

手法に応じた計算機インスタンスを生成します。
"""

from typing import Dict


from .base_calculator import BaseCalculator
from .fixed_quantity_calculator import FixedQuantityCalculator
from .fixed_ratio_calculator import FixedRatioCalculator
from .half_optimal_f_calculator import HalfOptimalFCalculator
from .volatility_based_calculator import VolatilityBasedCalculator


class CalculatorFactory:
    """計算機ファクトリ"""

    @staticmethod
    def create_calculator(method: str) -> BaseCalculator:
        """手法に応じた計算機インスタンスを生成"""
        method_map = {
            "half_optimal_f": HalfOptimalFCalculator,
            "volatility_based": VolatilityBasedCalculator,
            "fixed_ratio": FixedRatioCalculator,
            "fixed_quantity": FixedQuantityCalculator,
        }

        # enumからの変換
        if hasattr(method, "value"):
            method_str = method.value
        else:
            method_str = method

        calculator_class = method_map.get(method_str, FixedRatioCalculator)
        return calculator_class()

    @staticmethod
    def get_available_methods() -> Dict[str, str]:
        """利用可能な手法を取得"""
        return {
            "half_optimal_f": "ハーフオプティマルF",
            "volatility_based": "ボラティリティベース",
            "fixed_ratio": "固定比率",
            "fixed_quantity": "固定枚数",
        }
