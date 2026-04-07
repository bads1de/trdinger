"""
月初フィルター

毎月1日・2日など、月初の特異な資金フロー（給与、DCA、ファンドのリバランス）により
形成される特殊な相場環境（安値圏形成など）を回避するためのフィルターです。
"""

import random
from typing import Any, Dict

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool


class BeginningOfMonthFilter(BaseTool):
    """
    月初フィルター

    毎月1日から指定された日数間のエントリーをスキップします。
    """

    tool_definition = ToolDefinition(
        name="beginning_of_month_filter",
        description="月初の特異な需給バランスによる乱高下を回避します",
        default_params={"enabled": True, "days_from_start": 2},
    )

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """
        月初かどうかを判定

        Args:
            context: ツールコンテキスト
            params: パラメータ
                enabled (bool): 有効かどうか
                days_from_start (int): 月初から何日間をスキップするか
                                     1なら1日のみ、2なら1日と2日

        Returns:
            bool: スキップすべきならTrue
        """
        if not params.get("enabled", True) or context.timestamp is None:
            return False

        day = context.timestamp.day
        days_from_start = params.get("days_from_start", 2)

        # 指定日数以下ならスキップ
        # day == 1 <= 2 -> True
        # day == 2 <= 2 -> True
        # day == 3 <= 2 -> False
        return day <= days_from_start

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        パラメータを突然変異

        Args:
            params: 元のパラメータ

        Returns:
            変異後のパラメータ
        """
        new_params = super().mutate_params(params)

        # 20%の確率で期間を変更
        if random.random() < 0.2:
            current = new_params.get("days_from_start", 2)
            # 1〜5日の範囲で変動
            delta = random.randint(-1, 1)
            new_params["days_from_start"] = max(1, min(5, current + delta))

        return new_params


# グローバルインスタンスを作成してレジストリに登録
beginning_of_month_filter = BeginningOfMonthFilter()
register_tool(beginning_of_month_filter)
