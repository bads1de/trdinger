"""
計算機ファクトリ

手法に応じた計算機インスタンスを生成します。
"""

from enum import Enum
from typing import Any, Dict, Type, Union, cast

from ...utils.normalization import normalize_enum_name
from .base_calculator import BaseCalculator
from .fixed_quantity_calculator import FixedQuantityCalculator
from .fixed_ratio_calculator import FixedRatioCalculator
from .half_optimal_f_calculator import HalfOptimalFCalculator
from .volatility_based_calculator import VolatilityBasedCalculator


class CalculatorFactory:
    """計算機ファクトリ"""

    @staticmethod
    def create_calculator(method: Union[str, Enum]) -> BaseCalculator:
        """
        手法名に対応したポジションサイズ計算機インスタンスを生成

        Args:
            method: 計算方式名（'half_optimal_f', 'volatility_based' 等）

        Returns:
            BaseCalculatorを継承した計算機インスタンス
        """
        method_map = {
            "half_optimal_f": HalfOptimalFCalculator,
            "volatility_based": VolatilityBasedCalculator,
            "fixed_ratio": FixedRatioCalculator,
            "fixed_quantity": FixedQuantityCalculator,
        }

        # enumからの変換
        method_str = normalize_enum_name(method)

        calculator_class = cast(
            Type[BaseCalculator], method_map.get(method_str, FixedRatioCalculator)
        )
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
