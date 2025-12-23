"""
Adaptive Calculator

適応的TP/SL計算器
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ...genes import TPSLGene
from .base_calculator import BaseTPSLCalculator
from .fixed_percentage_calculator import FixedPercentageCalculator
from .risk_reward_calculator import RiskRewardCalculator
from .statistical_calculator import StatisticalCalculator
from .volatility_calculator import VolatilityCalculator

logger = logging.getLogger(__name__)


class AdaptiveCalculator(BaseTPSLCalculator):
    """
    適応的TP/SL計算器
    """

    def __init__(self):
        super().__init__("adaptive")
        self.calculators = {
            "fixed_percentage": FixedPercentageCalculator(),
            "risk_reward": RiskRewardCalculator(),
            "volatility": VolatilityCalculator(),
            "statistical": StatisticalCalculator(),
        }

    def _do_calculate(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene],
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
        **kwargs,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        # 1. 最適な計算方式を選択
        best_method = self._select_best_method(market_data, tpsl_gene)

        # 2. 選択された計算器で計算
        calculator = self.calculators[best_method]
        result = calculator.calculate(
            current_price=current_price,
            tpsl_gene=tpsl_gene,
            market_data=market_data,
            position_direction=position_direction,
            **kwargs,
        )

        return (
            result.stop_loss_pct,
            result.take_profit_pct,
            result.confidence_score,
            {**result.expected_performance, "adaptive_selection": best_method},
        )

    def _select_best_method(
        self,
        market_data: Optional[Dict[str, Any]],
        tpsl_gene: Optional[TPSLGene],
    ) -> str:
        """最適な計算方式を選択"""
        try:
            # 遺伝子での明示的な指定がある場合（ADAPTIVE以外）
            if tpsl_gene and tpsl_gene.method and tpsl_gene.method != "adaptive":
                # Enum または文字列からのマッピング
                method_name = (
                    tpsl_gene.method.value
                    if hasattr(tpsl_gene.method, "value")
                    else tpsl_gene.method
                )

                # 名称のゆらぎ吸収
                if method_name == "risk_reward_ratio":
                    method_name = "risk_reward"
                elif method_name == "volatility_based":
                    method_name = "volatility"

                if method_name in self.calculators:
                    return method_name

            if not market_data:
                return "fixed_percentage"

            # ボラティリティが高い場合
            volatility = market_data.get("volatility", "normal")
            if volatility in ["high", "very_high"]:
                return "volatility"

            # トレンドが明確な場合
            trend = market_data.get("trend", "neutral")
            if trend in ["strong_up", "strong_down"]:
                return "risk_reward"

            # 過去データが豊富な場合
            if market_data.get("historical_data_available", False):
                if len(market_data.get("historical_prices", [])) > 100:
                    return "statistical"

            # デフォルトは固定パーセンテージ
            return "fixed_percentage"

        except Exception as e:
            logger.error(f"最適方式選択エラー: {e}")
            return "fixed_percentage"
