"""
特定曜日フィルター

銘柄ごとに異なる「相性の悪い曜日」をスキップするための汎用フィルターです。
例: BTCは木・金曜のリターンが低い、STXは水曜が低いなどのアノマリーに対応します。
"""

import random
from typing import Any, Dict, List

from .base import BaseTool, ToolContext
from .registry import register_tool


class SpecificDayFilter(BaseTool):
    """
    特定曜日フィルター

    指定された曜日（0=月曜 〜 6=日曜）のエントリーをスキップします。
    """

    @property
    def name(self) -> str:
        return "specific_day_filter"

    @property
    def description(self) -> str:
        return "特定の曜日を指定してエントリーを回避します"

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """
        指定された曜日かどうかを判定

        Args:
            context: ツールコンテキスト
            params: パラメータ
                enabled (bool): 有効かどうか
                skip_days (List[int]): スキップする曜日のリスト (0=Mon, 6=Sun)

        Returns:
            bool: スキップすべきならTrue
        """
        if not params.get("enabled", True) or context.timestamp is None:
            return False

        # 現在の曜日を取得
        current_day = context.timestamp.dayofweek

        # スキップ対象リストを取得 (デフォルトは空)
        skip_days = params.get("skip_days", [])

        return current_day in skip_days

    def get_default_params(self) -> Dict[str, Any]:
        """
        デフォルトパラメータ

        Returns:
            enabled=True, skip_days=[]
        """
        return {"enabled": True, "skip_days": []}

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        パラメータを突然変異

        Args:
            params: 元のパラメータ

        Returns:
            変異後のパラメータ
        """
        new_params = super().mutate_params(params)
        skip_days = set(new_params.get("skip_days", []))

        # 30%の確率で曜日リストを変更
        if random.random() < 0.3:
            # ランダムな曜日 (0-6) を選ぶ
            day = random.randint(0, 6)

            if day in skip_days:
                skip_days.remove(day)  # 既に含まれていれば削除
            else:
                skip_days.add(day)  # 含まれていなければ追加

            new_params["skip_days"] = list(skip_days)

        return new_params


# グローバルインスタンスを作成してレジストリに登録
specific_day_filter = SpecificDayFilter()
register_tool(specific_day_filter)
