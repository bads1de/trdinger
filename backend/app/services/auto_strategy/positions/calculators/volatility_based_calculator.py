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

        # 1. 基本パラメータ取得
        risk_params = self._get_risk_params(gene)
        atr_value = market_data.get("atr", current_price * 0.02)
        atr_pct = atr_value / current_price if current_price > 0 else 0.02

        # 2. 基本ポジションサイズの計算
        risk_amount = account_balance * self._get_param(gene, "risk_per_trade", 0.02)
        volatility_factor = atr_pct * self._get_param(gene, "atr_multiplier", 2.0)
        
        if volatility_factor > 0:
            position_size = risk_amount / (current_price * volatility_factor)
        else:
            position_size = self._get_param(gene, "min_position_size", 0.001)
            warnings.append("ボラティリティが0、最小サイズを使用")

        # 3. リスク制限（VaR / Expected Shortfall）の適用
        raw_returns = market_data.get("returns")
        returns_data = list(raw_returns)[-risk_params["var_lookback"]:] if raw_returns is not None else []

        var_ratio = calculate_historical_var(returns_data, risk_params["var_confidence"])
        es_ratio = calculate_expected_shortfall(returns_data, risk_params["var_confidence"])

        price_for_adj = max(current_price, 1e-8)
        
        # VaR制限
        max_var = account_balance * risk_params["max_var_ratio"]
        if var_ratio > 0 and max_var > 0:
            if (position_size * price_for_adj * var_ratio) > max_var:
                position_size = max_var / (price_for_adj * var_ratio)
                warnings.append("VaR制限を適用")

        # ES制限
        max_es = account_balance * risk_params["max_es_ratio"]
        if es_ratio > 0 and max_es > 0:
            if (position_size * price_for_adj * es_ratio) > max_es:
                position_size = max_es / (price_for_adj * es_ratio)
                warnings.append("期待ショートフォール制限を適用")

        # 4. 詳細情報の構築
        details.update({
            "atr_pct": atr_pct,
            "risk_amount": risk_amount,
            "volatility_factor": volatility_factor,
            "var_ratio": var_ratio,
            "es_ratio": es_ratio,
            "return_sample_size": len(returns_data)
        })

        return self._apply_size_limits_and_finalize(position_size, details, warnings, gene)





