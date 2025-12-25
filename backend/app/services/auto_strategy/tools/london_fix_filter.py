"""
ロンドンフィックスフィルター

ロンドンフィックス（16:00 London Time）前後のボラティリティが高い時間帯のエントリーをスキップします。
機関投資家のリバランスフローによる予測困難な乱高下を回避します。
"""

import random
from typing import Any, Dict

from .base import BaseTool, ToolContext
from .registry import register_tool


class LondonFixFilter(BaseTool):
    """
    ロンドンフィックスフィルター

    ロンドンフィックス（16:00 London Time）前後のエントリーをスキップします。
    ロンドンフィックスは、夏時間はUTC 15:00、冬時間はUTC 16:00になります。
    デフォルトでは安全のため、両方の時間帯の前後をスキップ対象とします。
    """

    @property
    def name(self) -> str:
        return "london_fix_filter"

    @property
    def description(self) -> str:
        return "ロンドンフィックス（16:00 LDN）前後の乱高下を回避します"

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
        
        # 時間と分を取得
        hour = context.timestamp.hour
        minute = context.timestamp.minute
        
        window = params.get("window_minutes", 15)

        # ターゲット時間を分単位で計算
        current_minutes = hour * 60 + minute
        
        # 冬時間ターゲット (16:00 UTC) = 960分
        winter_target = 16 * 60
        # 夏時間ターゲット (15:00 UTC) = 900分
        summer_target = 15 * 60
        
        # 範囲内かチェック
        in_winter_fix = abs(current_minutes - winter_target) <= window
        in_summer_fix = abs(current_minutes - summer_target) <= window
        
        # どちらかの時間帯に入っていればスキップ
        return in_winter_fix or in_summer_fix

    def get_default_params(self) -> Dict[str, Any]:
        """
        デフォルトパラメータ

        Returns:
            enabled=True, window_minutes=15
        """
        return {
            "enabled": True,
            "window_minutes": 15
        }

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        パラメータを突然変異

        Args:
            params: 元のパラメータ

        Returns:
            変異後のパラメータ
        """
        new_params = params.copy()

        # 20%の確率で有効/無効を反転
        if random.random() < 0.2:
            new_params["enabled"] = not new_params.get("enabled", True)
            
        # 20%の確率でウィンドウサイズを変更 (5分〜30分)
        if random.random() < 0.2:
            current_window = new_params.get("window_minutes", 15)
            # -5〜+5分の範囲で変動、ただし5〜30分に収める
            delta = random.randint(-5, 5)
            new_window = max(5, min(30, current_window + delta))
            new_params["window_minutes"] = new_window

        return new_params


# グローバルインスタンスを作成してレジストリに登録
london_fix_filter = LondonFixFilter()
register_tool(london_fix_filter)
