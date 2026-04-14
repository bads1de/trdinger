"""
米国ランチタイムフィルター

米国東部時間（EST/EDT）のランチタイム（12:00-13:00）における流動性低下と
不規則な値動きを回避するためのフィルターです。
"""

import logging
from typing import Any, Dict

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool
from .time_windows import to_timezone_minutes

logger = logging.getLogger(__name__)


class USLunchFilter(BaseTool):
    """
    米国ランチタイムフィルター

    米国株式市場のランチタイム（12:00 - 13:00 EST/EDT）をスキップします。
    この時間帯は「エアポケット」と呼ばれ、流動性が落ちて価格形成が不安定になる傾向があります。

    EST (冬時間): UTC-5 -> ランチは UTC 17:00 - 18:00
    EDT (夏時間): UTC-4 -> ランチは UTC 16:00 - 17:00
    """

    tool_definition = ToolDefinition(
        name="us_lunch_filter",
        description="米国ランチタイム（12:00-13:00 EST）の流動性低下を回避します",
        default_params={"enabled": True},
    )

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """
        米国ランチタイムかどうかを判定

        Args:
            context: ツールコンテキスト
            params: パラメータ
                enabled (bool): 有効かどうか

        Returns:
            bool: スキップすべきならTrue
        """
        if not params.get("enabled", True) or context.timestamp is None:
            return False

        # タイムゾーン処理を行い、NY時間を取得
        try:
            current_minutes = to_timezone_minutes(context.timestamp, "US/Eastern")
            return current_minutes is not None and 12 * 60 <= current_minutes < 13 * 60

        except Exception as e:
            logger.debug(f"タイムゾーン変換に失敗しました（フォールバック適用）: {e}")
            from .time_windows import is_summer_time_by_month
            is_summer = is_summer_time_by_month(context.timestamp)
            hour = context.timestamp.hour

            if is_summer:
                return hour == 16
            else:
                return hour == 17


# グローバルインスタンスを作成してレジストリに登録
us_lunch_filter = USLunchFilter()
register_tool(us_lunch_filter)
