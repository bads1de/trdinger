"""
Risk Reward Calculator

リスクリワード比方式のTP/SL計算器
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ...genes import TPSLGene
from ...genes.tpsl import TPSLResult
from .base_calculator import BaseTPSLCalculator

logger = logging.getLogger(__name__)


class RiskRewardCalculator(BaseTPSLCalculator):
    """
    リスクリワード比方式のTP/SL計算器
    """

    def __init__(self):
        super().__init__("risk_reward_ratio")

    def _do_calculate(
        self, current_price: float, tpsl_gene: Optional[TPSLGene],
        market_data: Optional[Dict[str, Any]], position_direction: float, **kwargs
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        # 1. パラメータ取得
        if tpsl_gene:
            sl_pct = tpsl_gene.base_stop_loss or tpsl_gene.stop_loss_pct
            rr_ratio = tpsl_gene.risk_reward_ratio
        else:
            sl_pct = kwargs.get("stop_loss_pct", kwargs.get("base_stop_loss", 0.03))
            rr_ratio = kwargs.get("target_ratio", kwargs.get("risk_reward_ratio", 2.0))

        # 2. RR比に基づいてTPを計算
        tp_pct = sl_pct * rr_ratio

        return sl_pct, tp_pct, 0.85, {
            "risk_reward_ratio": rr_ratio,
            "base_stop_loss": sl_pct
        }





