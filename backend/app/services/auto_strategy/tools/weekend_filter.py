"""
週末フィルターツール

土曜日・日曜日のエントリーをスキップするフィルターです。
クリプト市場の週末は流動性が低く、ノイズが増える傾向があるため有効です。
"""

from typing import Any, Dict

from .base import BaseTool, ToolContext
from .registry import register_tool


class WeekendFilter(BaseTool):
    """
    週末フィルター

    土曜日(5)と日曜日(6)のエントリーをスキップします。
    実運用の知見から、週末トレード停止で成績が改善することが多いです。
    """

    @property
    def name(self) -> str:
        return "weekend_filter"

    @property
    def description(self) -> str:
        return "土曜日・日曜日のエントリーをスキップします"

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

    def get_default_params(self) -> Dict[str, Any]:
        """
        デフォルトパラメータ

        Returns:
            enabled=True の辞書
        """
        return {"enabled": True}


# グローバルインスタンスを作成してレジストリに登録
weekend_filter = WeekendFilter()
register_tool(weekend_filter)
