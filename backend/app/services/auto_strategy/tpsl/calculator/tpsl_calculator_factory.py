"""
TPSL計算機ファクトリ

手法に応じたTPSL計算機インスタンスを生成します。
"""

from typing import Any, Dict

from ...utils.normalization import normalize_enum_name
from .adaptive_calculator import AdaptiveCalculator
from .base_calculator import BaseTPSLCalculator
from .fixed_percentage_calculator import FixedPercentageCalculator
from .risk_reward_calculator import RiskRewardCalculator
from .statistical_calculator import StatisticalCalculator
from .volatility_calculator import VolatilityCalculator


class TPSLCalculatorFactory:
    """TPSL計算機ファクトリ"""

    # 計算機マッピング（クラスレベルで定義）
    _CALCULATORS: Dict[str, type[BaseTPSLCalculator]] = {
        "fixed_percentage": FixedPercentageCalculator,
        "risk_reward_ratio": RiskRewardCalculator,
        "volatility_based": VolatilityCalculator,
        "statistical": StatisticalCalculator,
        "adaptive": AdaptiveCalculator,
    }

    @classmethod
    def create_calculator(cls, method: Any) -> BaseTPSLCalculator:
        """
        手法名に対応したTPSL計算機インスタンスを生成

        Args:
            method: 計算方式名（'fixed_percentage', 'risk_reward_ratio' 等）

        Returns:
            BaseTPSLCalculatorを継承した計算機インスタンス

        Raises:
            ValueError: 未知の手法が指定された場合
        """
        # enumからの変換
        method_str = normalize_enum_name(method)

        calculator_class = cls._CALCULATORS.get(method_str)
        if calculator_class is None:
            # 未知の手法はエラーとして呼び出し元にフォールバックを委譲
            raise ValueError(f"未知のTPSL方式: {method_str}")

        return calculator_class()

    @classmethod
    def get_available_methods(cls) -> Dict[str, str]:
        """利用可能な手法を取得"""
        return {
            "fixed_percentage": "固定パーセンテージ",
            "risk_reward_ratio": "リスクリワード比率",
            "volatility_based": "ボラティリティベース",
            "statistical": "統計的",
            "adaptive": "アダプティブ",
        }

    @classmethod
    def register_calculator(
        cls, method_name: str, calculator_class: type[BaseTPSLCalculator]
    ) -> None:
        """
        カスタム計算機を登録

        Args:
            method_name: 方式名
            calculator_class: 計算機クラス
        """
        cls._CALCULATORS[method_name] = calculator_class
