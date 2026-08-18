"""
高ボラ停止フィルター（OHLCVだけで動く）

ATR%が閾値を超えた(高ボラ)時にエントリーをスキップ。
死んだvolatility_filterの価格版。extra_dataが無い場合はフェイルセーフで通す。
"""

import random
from typing import Any

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool


class VolatilityRegimeFilter(BaseTool):
    tool_definition = ToolDefinition(
        name="volatility_regime_filter",
        description="高ボラ時のエントリーを回避",
        default_params={"enabled": True, "atr_pct_threshold": 0.015},
        priority="disabled",
    )

    def should_skip_entry(self, context: ToolContext, params: dict[str, Any]) -> bool:
        if not params.get("enabled", True):
            return False
        atr_pct = context.extra_data.get("atr_pct")
        if atr_pct is None:
            atr = context.extra_data.get("atr")
            price = context.current_price
            if atr is not None and price:
                try:
                    atr_pct = float(atr) / float(price)
                except Exception:
                    return False
            else:
                return False
        try:
            thr = float(params.get("atr_pct_threshold", 0.015))
            return float(atr_pct) >= thr
        except Exception:
            return False

    def mutate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        new_params = super().mutate_params(params)
        if random.random() < 0.2:
            cur = float(new_params.get("atr_pct_threshold", 0.015))
            delta = random.uniform(-0.004, 0.004)
            new_params["atr_pct_threshold"] = max(0.005, min(0.04, cur + delta))
            new_params["atr_pct_threshold"] = round(new_params["atr_pct_threshold"], 4)
        return new_params


volatility_regime_filter = VolatilityRegimeFilter()
register_tool(volatility_regime_filter)
