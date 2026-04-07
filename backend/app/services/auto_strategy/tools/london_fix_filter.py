"""
ロンドンフィックスフィルター

ロンドンフィックス（16:00 London Time）前後のボラティリティが高い時間帯のエントリーをスキップします。
機関投資家のリバランスフローによる予測困難な乱高下を回避します。
"""

import random
from typing import Any, Dict

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool
from .time_windows import is_within_any_window, mutate_window_minutes, to_utc_minutes


class LondonFixFilter(BaseTool):
    """
    ロンドンフィックスフィルター

    ロンドンフィックス（16:00 London Time）前後のエントリーをスキップします。
    ロンドンフィックスは、夏時間はUTC 15:00、冬時間はUTC 16:00になります。
    デフォルトでは安全のため、両方の時間帯の前後をスキップ対象とします。
    """

    tool_definition = ToolDefinition(
        name="london_fix_filter",
        description="ロンドンフィックス（16:00 LDN）前後の乱高下を回避します",
        default_params={"enabled": True, "window_minutes": 15},
    )

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """
        ロンドンフィックス前後かどうかを判定

        Args:
            context: ツールコンテキスト
            params: パラメータ
                enabled (bool): 有効かどうか
                window_minutes (int): フィックス時間の前後何分をスキップするか（デフォルト15分）

        Returns:
            bool: スキップすべきならTrue
        """
        if not params.get("enabled", True) or context.timestamp is None:
            return False

        # timestampはUTCを想定
        # ロンドンフィックスは 16:00 London Time
        # Winter: UTC 16:00
        # Summer: UTC 15:00

        window = params.get("window_minutes", 15)
        current_minutes = to_utc_minutes(context.timestamp)
        return is_within_any_window(current_minutes, [15 * 60, 16 * 60], window)

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        パラメータを突然変異

        Args:
            params: 元のパラメータ

        Returns:
            変異後のパラメータ
        """
        new_params = super().mutate_params(params)

        # 20%の確率でウィンドウサイズを変更 (5分〜30分)
        if random.random() < 0.2:
            mutate_window_minutes(
                new_params,
                default=15,
                minimum=5,
                maximum=30,
                delta_low=-5,
                delta_high=5,
            )

        return new_params


# グローバルインスタンスを作成してレジストリに登録
london_fix_filter = LondonFixFilter()
register_tool(london_fix_filter)
