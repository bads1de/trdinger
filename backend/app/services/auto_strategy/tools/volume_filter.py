"""
出来高フィルター

現在の出来高が移動平均出来高を下回る場合、
流動性が低いと判断してエントリーを回避します。
"""

import logging
from typing import Any, Dict

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool

logger = logging.getLogger(__name__)


class VolumeFilter(BaseTool):
    """
    出来高フィルター

    現在の出来高が過去N期間の平均出来高に対して一定割合を下回る場合、
    流動性が低いと判断してエントリーをスキップします。

    出来高が低い市場では、スリッページが大きくなったり、
    価格形成が不安定になる傾向があります。

    パラメータ:
        min_volume_ratio: 最低出来高比率（現在出来高 / 平均出来高）。デフォルト0.5
        volume_period: 平均出来高の計算期間。デフォルト20
        enabled: 有効フラグ
    """

    tool_definition = ToolDefinition(
        name="volume_filter",
        description="出来高が平均より低い場合、流動性低下を回避してエントリーをスキップします",
        default_params={"enabled": True, "min_volume_ratio": 0.5, "volume_period": 20},
    )

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """
        出来高が低ければエントリーをスキップ

        Args:
            context: ツールコンテキスト
            params: パラメータ
                min_volume_ratio (float): 最低出来高比率
                enabled (bool): 有効かどうか

        Returns:
            bool: スキップすべきならTrue
        """
        if not params.get("enabled", True):
            return False

        min_volume_ratio = params.get("min_volume_ratio", 0.5)

        # extra_data に出来高関連データがあるかチェック
        current_volume = context.extra_data.get("current_volume", context.current_volume)
        avg_volume = context.extra_data.get("avg_volume")

        if current_volume is not None and avg_volume is not None and avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            return volume_ratio < min_volume_ratio

        # current_volumeがcontextから取れる場合（extra_dataがない場合）
        if context.current_volume > 0 and avg_volume is not None and avg_volume > 0:
            volume_ratio = context.current_volume / avg_volume
            return volume_ratio < min_volume_ratio

        # データがない場合はスキップしない
        return False

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータの突然変異"""
        import random

        new_params = super().mutate_params(params)

        if random.random() < 0.3:
            new_params["min_volume_ratio"] = max(
                0.1, min(1.5, new_params.get("min_volume_ratio", 0.5) * random.uniform(0.8, 1.2))
            )
        if random.random() < 0.2:
            new_params["volume_period"] = max(
                5, min(60, int(new_params.get("volume_period", 20) * random.uniform(0.8, 1.2)))
            )

        return new_params


volume_filter = VolumeFilter()
register_tool(volume_filter)
