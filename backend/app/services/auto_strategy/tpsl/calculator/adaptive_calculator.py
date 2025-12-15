"""
Adaptive Calculator

適応的TP/SL計算器
"""

import logging
from typing import Any, Dict, Optional

from ...genes import TPSLGene
from ...genes.tpsl import TPSLResult
from .base_calculator import BaseTPSLCalculator
from .fixed_percentage_calculator import FixedPercentageCalculator
from .risk_reward_calculator import RiskRewardCalculator
from .statistical_calculator import StatisticalCalculator
from .volatility_calculator import VolatilityCalculator

logger = logging.getLogger(__name__)


class AdaptiveCalculator(BaseTPSLCalculator):
    """
    適応的TP/SL計算器

    市場条件に基づいて自動的に最適な計算方式を選択します。
    """

    def __init__(self):
        """初期化"""
        super().__init__("adaptive")
        self.calculators = {
            "fixed_percentage": FixedPercentageCalculator(),
            "risk_reward": RiskRewardCalculator(),
            "volatility": VolatilityCalculator(),
            "statistical": StatisticalCalculator(),
        }

    def calculate(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene] = None,
        market_data: Optional[Dict[str, Any]] = None,
        position_direction: float = 1.0,
        **kwargs,
    ) -> TPSLResult:
        """
        適応的にTP/SLを計算

        Args:
            current_price: 現在価格
            tpsl_gene: TP/SL遺伝子
            market_data: 市場データ
            position_direction: ポジション方向（1.0=ロング, -1.0=ショート）
            **kwargs: 追加パラメータ

        Returns:
            TPSLResult: 計算結果
        """
        try:
            # 最適な計算方式を選択
            best_method = self._select_best_method(market_data, tpsl_gene)

            # 選択された計算器で計算
            calculator = self.calculators[best_method]
            result = calculator.calculate(
                current_price=current_price,
                tpsl_gene=tpsl_gene,
                market_data=market_data,
                position_direction=position_direction,
                **kwargs,
            )

            # 適応的選択の情報を追加
            result.expected_performance["adaptive_selection"] = best_method
            result.metadata["selected_method"] = best_method

            return result

        except Exception as e:
            logger.error(f"適応的計算エラー: {e}")
            # フォールバック
            fallback_calculator = FixedPercentageCalculator()
            return fallback_calculator.calculate(
                current_price=current_price,
                tpsl_gene=tpsl_gene,
                market_data=market_data,
                position_direction=position_direction,
                **kwargs,
            )

    def _select_best_method(
        self,
        market_data: Optional[Dict[str, Any]],
        tpsl_gene: Optional[TPSLGene],
    ) -> str:
        """最適な計算方式を選択"""
        try:
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

            # TPSL遺伝子がある場合、そのmethodを尊重
            if tpsl_gene and hasattr(tpsl_gene, "method"):
                method_name = tpsl_gene.method.name.lower()
                if method_name in self.calculators:
                    return method_name

            # デフォルトは固定パーセンテージ
            return "fixed_percentage"

        except Exception as e:
            logger.error(f"最適方式選択エラー: {e}")
            return "fixed_percentage"





