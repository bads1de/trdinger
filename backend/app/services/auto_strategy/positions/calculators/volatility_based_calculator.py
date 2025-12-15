"""
ボラティリティベース方式計算クラス
"""

import logging
from typing import Any, Dict, List

from ..risk_metrics import (
    calculate_expected_shortfall,
    calculate_historical_var,
)
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

        returns_data: List[float] = []
        raw_returns = market_data.get("returns")
        if raw_returns is not None:
            try:
                returns_list = list(raw_returns)
                lookback = max(int(getattr(gene, "var_lookback", len(returns_list))), 1)
                returns_data = returns_list[-lookback:]
            except TypeError:
                returns_data = []

        var_ratio = calculate_historical_var(
            returns_data, getattr(gene, "var_confidence", 0.95)
        )
        expected_shortfall_ratio = calculate_expected_shortfall(
            returns_data, getattr(gene, "var_confidence", 0.95)
        )

        price_for_adjustment = current_price if current_price > 0 else 1e-8
        position_value = position_size * price_for_adjustment

        risk_controls = {
            "var_ratio": var_ratio,
            "var_confidence": getattr(gene, "var_confidence", 0.95),
            "var_lookback": getattr(gene, "var_lookback", len(returns_data)),
            "max_var_allowed": account_balance * getattr(gene, "max_var_ratio", 0.0),
            "var_loss": position_value * var_ratio,
            "var_adjusted": False,
            "expected_shortfall": expected_shortfall_ratio,
            "max_expected_shortfall_allowed": account_balance
            * getattr(gene, "max_expected_shortfall_ratio", 0.0),
            "expected_shortfall_loss": position_value * expected_shortfall_ratio,
            "expected_shortfall_adjusted": False,
            "return_sample_size": len(returns_data),
        }

        if var_ratio > 0 and risk_controls["max_var_allowed"] > 0:
            if risk_controls["var_loss"] > risk_controls["max_var_allowed"]:
                capped_value = risk_controls["max_var_allowed"] / max(var_ratio, 1e-12)
                position_value = max(
                    capped_value, gene.min_position_size * price_for_adjustment
                )
                position_size = position_value / price_for_adjustment
                risk_controls["var_adjusted"] = True
                warnings.append("VaR制限を適用しました")
                risk_controls["var_loss"] = position_value * var_ratio
                risk_controls["expected_shortfall_loss"] = (
                    position_value * expected_shortfall_ratio
                )

        if (
            expected_shortfall_ratio > 0
            and risk_controls["max_expected_shortfall_allowed"] > 0
        ):
            current_es_loss = (
                position_size * price_for_adjustment * expected_shortfall_ratio
            )
            if current_es_loss > risk_controls["max_expected_shortfall_allowed"]:
                capped_value = risk_controls["max_expected_shortfall_allowed"] / max(
                    expected_shortfall_ratio, 1e-12
                )
                position_value = max(
                    capped_value, gene.min_position_size * price_for_adjustment
                )
                position_size = position_value / price_for_adjustment
                risk_controls["expected_shortfall_adjusted"] = True
                warnings.append("期待ショートフォール制限を適用しました")
                risk_controls["var_loss"] = position_value * var_ratio
                risk_controls["expected_shortfall_loss"] = (
                    position_value * expected_shortfall_ratio
                )

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

        details["risk_controls"] = risk_controls

        # 最大サイズ制限適用 + 統一された最終処理
        position_size = min(position_size, gene.max_position_size)
        return self._apply_size_limits_and_finalize(
            position_size, details, warnings, gene
        )





