"""
ボラティリティベース方式計算クラス
"""

import logging
from typing import Any, Dict
from .base_calculator import BaseCalculator

logger = logging.getLogger(__name__)


class VolatilityBasedCalculator(BaseCalculator):
    """ボラティリティベース方式計算クラス"""

    def calculate(
        self, gene, account_balance: float, current_price: float, **kwargs
    ) -> Dict[str, Any]:
        """ボラティリティベース方式の拡張計算"""
        market_data = kwargs.get("market_data", {})
        details: Dict[str, Any] = {"method": "volatility_based"}
        warnings = []

        # ATR値の取得
        atr_value = market_data.get("atr", current_price * 0.02)
        atr_pct = atr_value / current_price if current_price > 0 else 0.02

        # リスク量の計算
        risk_amount = account_balance * gene.risk_per_trade

        # ポジションサイズの計算
        volatility_factor = atr_pct * gene.atr_multiplier
        if volatility_factor > 0:
            position_size = risk_amount / (current_price * volatility_factor)
        else:
            position_size = gene.min_position_size
            warnings.append("ボラティリティが0、最小サイズを使用")

        # 詳細情報の更新
        details.update(
            {
                "atr_value": atr_value,
                "atr_pct": atr_pct,
                "atr_multiplier": gene.atr_multiplier,
                "risk_per_trade": gene.risk_per_trade,
                "risk_amount": risk_amount,
                "volatility_factor": volatility_factor,
                "atr_source": market_data.get("atr_source", "provided"),
            }
        )

        # 最大サイズ制限適用 + 統一された最終処理
        position_size = min(position_size, gene.max_position_size)
        return self._apply_size_limits_and_finalize(
            position_size, details, warnings, gene
        )
