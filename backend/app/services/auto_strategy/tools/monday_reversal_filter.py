"""
月曜反転フィルター

「日曜高・月曜安」のアノマリーに基づき、週明け月曜日の特定時間帯のエントリーをスキップします。
週末の薄商いで上昇した分が、機関投資家が戻ってくる月曜日に調整（Reversal）されるリスクを回避します。
"""

import random
from typing import Any, Dict

from .base import BaseTool, ToolContext
from .registry import register_tool


class MondayReversalFilter(BaseTool):
    """
    月曜反転フィルター

    月曜日（週明け）の開始から一定時間のエントリーをスキップします。
    デフォルトでは月曜日の最初の12時間（UTC 00:00 - 12:00）を回避します。
    """

    @property
    def name(self) -> str:
        return "monday_reversal_filter"

    @property
    def description(self) -> str:
        return "月曜日前半の調整局面（Reversal）を回避します"

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """
        月曜日の指定時間帯かどうかを判定

        Args:
            context: ツールコンテキスト
            params: パラメータ
                enabled (bool): 有効かどうか
                skip_hours (int): 月曜開始から何時間をスキップするか（デフォルト12）

        Returns:
            bool: スキップすべきならTrue
        """
        if not params.get("enabled", True) or context.timestamp is None:
            return False

        # 曜日を取得（0=Mon, 5=Sat, 6=Sun）
        # context.timestamp は pd.Timestamp を想定
        if context.timestamp.dayofweek != 0:  # 月曜日以外はスキップしない
            return False

        # 月曜日の場合、時間をチェック
        hour = context.timestamp.hour
        skip_hours = params.get("skip_hours", 12)

        # 指定時間未満ならスキップ (例: skip_hours=12なら 00:00〜11:59 までスキップ)
        return hour < skip_hours

    def get_default_params(self) -> Dict[str, Any]:
        """
        デフォルトパラメータ

        Returns:
            enabled=True, skip_hours=12
        """
        return {"enabled": True, "skip_hours": 12}

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        パラメータを突然変異

        Args:
            params: 元のパラメータ

        Returns:
            変異後のパラメータ
        """
        new_params = super().mutate_params(params)

        # 20%の確率でスキップ時間を変更 (4時間〜20時間)
        if random.random() < 0.2:
            current_hours = new_params.get("skip_hours", 12)
            # -4〜+4時間の範囲で変動
            delta = random.randint(-4, 4)
            new_hours = max(4, min(20, current_hours + delta))
            new_params["skip_hours"] = new_hours

        return new_params


# グローバルインスタンスを作成してレジストリに登録
monday_reversal_filter = MondayReversalFilter()
register_tool(monday_reversal_filter)
