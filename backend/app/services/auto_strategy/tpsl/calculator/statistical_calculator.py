"""
Statistical Calculator

統計的分析ベースのTP/SL計算器
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ...genes import TPSLGene
from .base_calculator import BaseTPSLCalculator

logger = logging.getLogger(__name__)


class StatisticalCalculator(BaseTPSLCalculator):
    """
    統計的分析ベースのTP/SL計算器
    """

    def __init__(self):
        super().__init__("statistical")

    def _do_calculate(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene],
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
        **kwargs,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        # 1. パラメータ取得
        if tpsl_gene:
            lookback_period = tpsl_gene.lookback_period or 150
            confidence_threshold = tpsl_gene.confidence_threshold or 0.95
        else:
            lookback_period = kwargs.get(
                "lookback_period_days", kwargs.get("lookback_period", 150)
            )
            confidence_threshold = kwargs.get("confidence_threshold", 0.95)

        # 2. 統計分析で最適なTP/SLを計算
        sl_pct, tp_pct = self._calculate_statistical_levels(
            market_data, lookback_period, confidence_threshold, current_price
        )

        return (
            sl_pct,
            tp_pct,
            0.7,
            {
                "lookback_period": lookback_period,
                "confidence_threshold": confidence_threshold,
            },
        )

    def _calculate_statistical_levels(
        self,
        market_data: Optional[Dict[str, Any]],
        lookback_period: int,
        confidence_threshold: float,
        current_price: float,
    ) -> Tuple[float, float]:
        """統計的な最適レベルを計算"""
        try:
            if not market_data or "historical_prices" not in market_data:
                # データがない場合のデフォルト値
                return 0.03, 0.06

            historical_prices = market_data["historical_prices"]
            if len(historical_prices) < lookback_period:
                return 0.03, 0.06

            # 過去データの統計分析
            recent_prices = historical_prices[-lookback_period:]

            # 価格変化の分布を計算
            price_changes = []
            for i in range(1, len(recent_prices)):
                change_pct = (recent_prices[i] - recent_prices[i - 1]) / recent_prices[
                    i - 1
                ]
                price_changes.append(change_pct)

            if not price_changes:
                return 0.03, 0.06

            # 標準偏差を計算
            mean_change = sum(price_changes) / len(price_changes)
            variance = sum((x - mean_change) ** 2 for x in price_changes) / len(
                price_changes
            )
            std_dev = variance**0.5

            # 信頼区間に基づいてSL/TPを設定
            # 95%信頼区間を使用（正規分布を仮定）
            z_score = 1.96  # 95%信頼区間

            sl_std_devs = z_score
            tp_std_devs = z_score * 2  # TPはSLの2倍の変動を想定

            stop_loss_pct = abs(std_dev * sl_std_devs)
            take_profit_pct = abs(std_dev * tp_std_devs)

            # 最小/最大値の制約
            stop_loss_pct = max(0.01, min(stop_loss_pct, 0.1))  # 1% - 10%
            take_profit_pct = max(0.02, min(take_profit_pct, 0.2))  # 2% - 20%

            return stop_loss_pct, take_profit_pct

        except Exception as e:
            logger.error(f"統計レベル計算エラー: {e}")
            return 0.03, 0.06
