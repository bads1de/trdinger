"""
ボラティリティフィルター

ATR（Average True Range）またはボラティリティが閾値以下の場合に
エントリーをスキップします。流動性が低い、または価格変動が小さい
環境でのエントリーを回避します。
"""

import logging
from typing import Any, Dict

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool

logger = logging.getLogger(__name__)


class VolatilityFilter(BaseTool):
    """
    ボラティリティフィルター

    現在のボラティリティが最低閾値を下回る場合、エントリーをスキップします。
    これにより、流動性が低く価格変動が小さい環境でのエントリーを回避できます。

    パラメータ:
        min_atr_pct: 最低ATR変化率（現在価格に対するATRの割合）。デフォルト0.001（0.1%）
        atr_period: ATR計算期間。デフォルト14
        enabled: 有効フラグ
    """

    tool_definition = ToolDefinition(
        name="volatility_filter",
        description="ボラティリティが低い環境でのエントリーを回避します",
        default_params={"enabled": True, "min_atr_pct": 0.001, "atr_period": 14},
    )

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """
        ボラティリティが低ければエントリーをスキップ

        Args:
            context: ツールコンテキスト
            params: パラメータ
                min_atr_pct (float): 最低ATR変化率
                enabled (bool): 有効かどうか

        Returns:
            bool: スキップすべきならTrue
        """
        if not params.get("enabled", True):
            return False

        min_atr_pct = params.get("min_atr_pct", 0.001)

        # extra_data にATR値が渡されているかチェック
        atr_value = context.extra_data.get("atr")
        if atr_value is not None and atr_value > 0:
            atr_pct = atr_value / max(context.current_price, 1e-12)
            return atr_pct < min_atr_pct

        # ATRがない場合、extra_dataのvolatility（標準偏差）で判定
        volatility = context.extra_data.get("volatility")
        if volatility is not None and volatility > 0:
            vol_pct = volatility / max(context.current_price, 1e-12)
            return vol_pct < min_atr_pct

        # データがない場合はスキップしない（安全側に倒す）
        return False

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータの突然変異"""
        import random

        new_params = super().mutate_params(params)

        if random.random() < 0.3:
            new_params["min_atr_pct"] = max(
                0.0001,
                min(
                    0.01,
                    new_params.get("min_atr_pct", 0.001) * random.uniform(0.8, 1.2),
                ),
            )
        if random.random() < 0.2:
            new_params["atr_period"] = max(
                5,
                min(
                    50, int(new_params.get("atr_period", 14) * random.uniform(0.8, 1.2))
                ),
            )

        return new_params


volatility_filter = VolatilityFilter()
register_tool(volatility_filter)
