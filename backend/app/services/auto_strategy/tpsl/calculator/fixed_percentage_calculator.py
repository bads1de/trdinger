"""
Fixed Percentage Calculator

固定パーセンテージ方式のTP/SL計算器
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ...genes import TPSLGene
from ...genes.tpsl import TPSLResult
from .base_calculator import BaseTPSLCalculator

logger = logging.getLogger(__name__)


class FixedPercentageCalculator(BaseTPSLCalculator):
    """
    固定パーセンテージ方式のTP/SL計算器
    """

    def __init__(self):
        super().__init__("fixed_percentage")

    def _do_calculate(
        self, current_price: float, tpsl_gene: Optional[TPSLGene],
        market_data: Optional[Dict[str, Any]], position_direction: float, **kwargs
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        # 1. パラメータ取得
        if tpsl_gene:
            sl_pct = tpsl_gene.stop_loss_pct
            tp_pct = tpsl_gene.take_profit_pct
        else:
            sl_pct = kwargs.get("stop_loss_pct", 0.03)
            tp_pct = kwargs.get("take_profit_pct", 0.06)

        # 2. 価格生成（デバッグ・期待値用）
        sl_price, tp_price = self._make_prices(current_price, sl_pct, tp_pct, position_direction)

        return sl_pct, tp_pct, 0.95, {
            "sl_price": sl_price,
            "tp_price": tp_price
        }





