"""
米国市場オープンフィルター

米国株式市場の開始（09:30 EST/EDT）前後の高ボラティリティを回避するためのフィルターです。
ETFフローやマクロニュースの織り込みにより、最も相場が乱高下しやすい時間帯です。
"""

import logging
import random
from typing import Any, Dict

from .base import BaseTool, ToolContext, ToolDefinition
from .registry import register_tool
from .time_windows import (
    is_within_window,
    mutate_window_minutes,
    to_timezone_minutes,
)

logger = logging.getLogger(__name__)


class USMarketOpenFilter(BaseTool):
    """
    米国市場オープンフィルター

    米国株式市場のオープニング（09:30 EST/EDT）前後をスキップします。

    EST (冬時間): UTC-5 -> Open UTC 14:30
    EDT (夏時間): UTC-4 -> Open UTC 13:30
    """

    tool_definition = ToolDefinition(
        name="us_market_open_filter",
        description="米国市場開始（09:30 EST）前後の乱高下を回避します",
        default_params={"enabled": True, "window_minutes": 30},
    )

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        """
        米国市場オープン前後かどうかを判定

        Args:
            context: ツールコンテキスト
            params: パラメータ
                enabled (bool): 有効かどうか
                window_minutes (int): 開始時間の前後何分をスキップするか（デフォルト30分）

        Returns:
            bool: スキップすべきならTrue
        """
        if not params.get("enabled", True) or context.timestamp is None:
            return False

        try:
            current_minutes = to_timezone_minutes(context.timestamp, "US/Eastern")
            open_minutes = 9 * 60 + 30  # 09:30 = 570分

            window = params.get("window_minutes", 30)

            # 前後 window 分の範囲内ならスキップ
            return is_within_window(current_minutes, open_minutes, window)

        except Exception as e:
            logger.debug(f"タイムゾーン変換に失敗しました（フォールバック適用）: {e}")
            from .time_windows import is_summer_time_by_month
            is_summer = is_summer_time_by_month(context.timestamp)

            target_hour = 13 if is_summer else 14
            target_minute = 30

            ts_hour = context.timestamp.hour
            ts_minute = context.timestamp.minute

            current_min = ts_hour * 60 + ts_minute
            target_min = target_hour * 60 + target_minute

            window = params.get("window_minutes", 30)

            return abs(current_min - target_min) <= window

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        パラメータを突然変異

        Args:
            params: 元のパラメータ

        Returns:
            変異後のパラメータ
        """
        new_params = super().mutate_params(params)

        # 20%の確率でウィンドウサイズを変更
        if random.random() < 0.2:
            mutate_window_minutes(
                new_params,
                default=30,
                minimum=10,
                maximum=60,
                delta_low=-10,
                delta_high=10,
            )

        return new_params


# グローバルインスタンスを作成してレジストリに登録
us_market_open_filter = USMarketOpenFilter()
register_tool(us_market_open_filter)
