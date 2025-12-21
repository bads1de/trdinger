"""
Volatility Calculator

ボラティリティベースのTP/SL計算器
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ...genes import TPSLGene
from .base_calculator import BaseTPSLCalculator

logger = logging.getLogger(__name__)


class VolatilityCalculator(BaseTPSLCalculator):
    """
    ボラティリティベースのTP/SL計算器
    """

    def __init__(self):
        super().__init__("volatility_based")

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
            atr_period = tpsl_gene.atr_period or 21
            atr_multiplier_sl = tpsl_gene.atr_multiplier_sl or 1.5
            atr_multiplier_tp = tpsl_gene.atr_multiplier_tp or 3.0
        else:
            atr_period = kwargs.get("atr_period", 21)
            atr_multiplier_sl = kwargs.get("atr_multiplier_sl", 1.5)
            atr_multiplier_tp = kwargs.get("atr_multiplier_tp", 3.0)

        # 2. ATR値を取得（または計算）
        atr_value = self._get_atr_value(market_data, atr_period, current_price)

        # 3. ATRベースの割合を計算
        base_atr_pct = atr_value / current_price if atr_value else 0.02
        sl_pct = base_atr_pct * atr_multiplier_sl
        tp_pct = base_atr_pct * atr_multiplier_tp

        return (
            sl_pct,
            tp_pct,
            0.9,
            {
                "atr_period": atr_period,
                "atr_multiplier_sl": atr_multiplier_sl,
                "atr_multiplier_tp": atr_multiplier_tp,
                "atr_value": atr_value,
                "base_atr_pct": base_atr_pct,
            },
        )

    def _get_atr_value(
        self,
        market_data: Optional[Dict[str, Any]],
        atr_period: int,
        current_price: float,
    ) -> Optional[float]:
        """ATR値を取得または計算"""
        if not market_data:
            return None

        # 直接ATR値が提供されている場合
        if "atr" in market_data:
            return market_data["atr"]

        # OHLCデータからATRを計算
        if "ohlc_data" in market_data:
            return self._calculate_atr_from_ohlc(market_data["ohlc_data"], atr_period)

        # ボラティリティから推定
        if "volatility" in market_data:
            return current_price * market_data["volatility"]

        return None

    def _calculate_atr_from_ohlc(
        self, ohlc_data: list, atr_period: int
    ) -> Optional[float]:
        """OHLCデータからATRを計算"""
        try:
            if len(ohlc_data) < atr_period:
                return None

            true_ranges = []
            for i in range(1, len(ohlc_data)):
                high = ohlc_data[i]["high"]
                low = ohlc_data[i]["low"]
                prev_close = ohlc_data[i - 1]["close"]

                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                true_ranges.append(tr)

            # ATR = 平均True Range
            atr = sum(true_ranges[-atr_period:]) / atr_period
            return atr

        except Exception as e:
            logger.error(f"ATR計算エラー: {e}")
            return None
