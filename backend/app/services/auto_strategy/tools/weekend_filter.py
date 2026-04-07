"""
週末フィルターツール

土曜日・日曜日のエントリーをスキップするフィルターです。
クリプト市場の週末は流動性が低く、ノイズが増える傾向があるため有効です。
"""

from typing import Any, Dict

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool


class WeekendFilter(BaseTool):
    """
    週末フィルター

    土曜日(5)と日曜日(6)のエントリーをスキップします。
    実運用の知見から、週末トレード停止で成績が改善することが多いです。
    """

    tool_definition = ToolDefinition(
        name="weekend_filter",
        description="土曜日・日曜日のエントリーをスキップします",
        default_params={"enabled": True},
    )

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """週末かどうかを判定してエントリースキップを決定"""
        if not params.get("enabled", True) or context.timestamp is None:
            return False

        # 曜日を取得（0=Mon, 5=Sat, 6=Sun）
        try:
            # context.timestamp が pd.Timestamp であることを想定
            weekday = context.timestamp.weekday()
            return weekday in (5, 6)
        except (AttributeError, Exception):
            return False


# グローバルインスタンスを作成してレジストリに登録
weekend_filter = WeekendFilter()
register_tool(weekend_filter)
