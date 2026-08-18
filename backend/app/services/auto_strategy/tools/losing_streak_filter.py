"""
連敗停止フィルター

直近N連敗したら次のエントリーをスキップする。
負けが続くレンジで無駄打ちを防ぐ。
"""

import random
from typing import Any

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool


class LosingStreakFilter(BaseTool):
    tool_definition = ToolDefinition(
        name="losing_streak_filter",
        description="連敗時のエントリーを回避",
        default_params={"enabled": True, "consecutive_losses": 3, "cooldown_bars": 5},
        priority="disabled",
    )

    def should_skip_entry(self, context: ToolContext, params: dict[str, Any]) -> bool:
        if not params.get("enabled", True):
            return False
        trades = context.extra_data.get("closed_trades")
        if not trades:
            return False
        try:
            n = int(params.get("consecutive_losses", 3))
        except Exception:
            n = 3
        n = max(2, min(5, n))
        if len(trades) < n:
            return False
        # last n trades
        recent = trades[-n:]
        losses = 0
        for t in recent:
            # support dict or object
            val = None
            if isinstance(t, dict):
                val = t.get("pl_pct", t.get("return_pct", t.get("pl", t.get("pnl"))))
            else:
                for attr in ("pl_pct", "return_pct", "pl", "pnl"):
                    v = getattr(t, attr, None)
                    if v is not None:
                        val = v
                        break
            if val is None:
                return False
            try:
                if float(val) < 0:
                    losses += 1
                else:
                    return False
            except Exception:
                return False
        if losses < n:
            return False
        # optional cooldown: check bars_since_last_trade if available
        # if cooldown_bars provided, we rely on caller to manage; simple version: always skip when streak
        return True

    def mutate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        new_params = super().mutate_params(params)
        if random.random() < 0.2:
            cur = int(new_params.get("consecutive_losses", 3))
            delta = random.randint(-1, 1)
            new_params["consecutive_losses"] = max(2, min(5, cur + delta))
        if random.random() < 0.2:
            cur = int(new_params.get("cooldown_bars", 5))
            delta = random.randint(-2, 2)
            new_params["cooldown_bars"] = max(3, min(20, cur + delta))
        return new_params


losing_streak_filter = LosingStreakFilter()
register_tool(losing_streak_filter)
