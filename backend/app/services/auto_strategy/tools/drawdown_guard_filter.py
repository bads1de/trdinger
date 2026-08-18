"""
ドローダウンガードフィルター

直近のドローダウンが閾値を超えた場合にエントリーを停止する。
連続損失やトレンド崩れ時のエントリーを抑制し、WFA落ちを防ぐ。
"""

import random
from typing import Any

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool


class DrawdownGuardFilter(BaseTool):
    tool_definition = ToolDefinition(
        name="drawdown_guard_filter",
        description="ドローダウン超過時のエントリーを回避",
        default_params={"enabled": True, "threshold": 0.05},
        priority="disabled",
    )

    def should_skip_entry(self, context: ToolContext, params: dict[str, Any]) -> bool:
        if not params.get("enabled", True):
            return False
        equity = context.extra_data.get("equity")
        max_equity = context.extra_data.get("max_equity")
        if equity is None or max_equity is None:
            return False
        try:
            eq = float(equity)
            mx = float(max_equity)
        except Exception:
            return False
        if mx <= 0:
            return False
        dd = (mx - eq) / mx
        threshold = float(params.get("threshold", 0.05))
        return dd >= threshold

    def mutate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        new_params = super().mutate_params(params)
        if random.random() < 0.2:
            cur = float(new_params.get("threshold", 0.05))
            delta = random.uniform(-0.015, 0.015)
            new_params["threshold"] = max(0.02, min(0.10, cur + delta))
            # round to 3 decimals
            new_params["threshold"] = round(new_params["threshold"], 3)
        return new_params


drawdown_guard_filter = DrawdownGuardFilter()
register_tool(drawdown_guard_filter)
