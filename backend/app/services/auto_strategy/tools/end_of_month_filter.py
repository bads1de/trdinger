"""
月末フィルター

月末の機関投資家によるリバランスやウィンドウドレッシングに起因する
不規則な価格変動を回避するためのフィルターです。
"""

import random
import calendar
from typing import Any, Dict

from .base import BaseTool, ToolContext
from .registry import register_tool


class EndOfMonthFilter(BaseTool):
    """
    月末フィルター

    月の最終日（およびその数日前）のエントリーをスキップします。
    """

    @property
    def name(self) -> str:
        return "end_of_month_filter"

    @property
    def description(self) -> str:
        return "月末のリバランスによる不規則な動きを回避します"

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """
        月末かどうかを判定

        Args:
            context: ツールコンテキスト
            params: パラメータ
                enabled (bool): 有効かどうか
                days_before_end (int): 月末から何日前まで遡ってスキップするか
                                     0なら最終日のみ、1なら最終日と前日

        Returns:
            bool: スキップすべきならTrue
        """
        if not params.get("enabled", True) or context.timestamp is None:
            return False

        ts = context.timestamp
        year = ts.year
        month = ts.month
        day = ts.day

        # その月の日数（最終日）を取得
        _, last_day = calendar.monthrange(year, month)

        days_before = params.get("days_before_end", 0)

        # 月末までの残り日数
        days_left = last_day - day

        # 残り日数が指定範囲内ならスキップ
        # days_left == 0 (最終日) <= 0 -> True
        return days_left <= days_before

    def get_default_params(self) -> Dict[str, Any]:
        """
        デフォルトパラメータ

        Returns:
            enabled=True, days_before_end=0 (最終日のみ)
        """
        return {"enabled": True, "days_before_end": 0}

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        パラメータを突然変異

        Args:
            params: 元のパラメータ

        Returns:
            変異後のパラメータ
        """
        new_params = super().mutate_params(params)

        # 20%の確率で対象期間を変更
        if random.random() < 0.2:
            current = new_params.get("days_before_end", 0)
            # 0(最終日のみ) 〜 3(ラスト4日間) の範囲で変動
            delta = random.randint(-1, 1)
            new_params["days_before_end"] = max(0, min(3, current + delta))

        return new_params


# グローバルインスタンスを作成してレジストリに登録
end_of_month_filter = EndOfMonthFilter()
register_tool(end_of_month_filter)
