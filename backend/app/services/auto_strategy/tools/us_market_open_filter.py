"""
米国市場オープンフィルター

米国株式市場の開始（09:30 EST/EDT）前後の高ボラティリティを回避するためのフィルターです。
ETFフローやマクロニュースの織り込みにより、最も相場が乱高下しやすい時間帯です。
"""

import random
from typing import Any, Dict
import pandas as pd

from .base import BaseTool, ToolContext
from .registry import register_tool


class USMarketOpenFilter(BaseTool):
    """
    米国市場オープンフィルター

    米国株式市場のオープニング（09:30 EST/EDT）前後をスキップします。
    
    EST (冬時間): UTC-5 -> Open UTC 14:30
    EDT (夏時間): UTC-4 -> Open UTC 13:30
    """

    @property
    def name(self) -> str:
        return "us_market_open_filter"

    @property
    def description(self) -> str:
        return "米国市場開始（09:30 EST）前後の乱高下を回避します"

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
            # timestampがnaiveな場合、UTCとみなして変換
            ts = context.timestamp
            if ts.tz is None:
                ts = ts.tz_localize('UTC')
            
            # NY時間に変換
            ny_time = ts.tz_convert('US/Eastern')
            
            # 市場開始は 09:30
            # 分単位で計算
            current_minutes = ny_time.hour * 60 + ny_time.minute
            open_minutes = 9 * 60 + 30  # 09:30 = 570分
            
            window = params.get("window_minutes", 30)
            
            # 前後 window 分の範囲内ならスキップ
            return abs(current_minutes - open_minutes) <= window
            
        except Exception:
            # 変換失敗時は簡易判定（UTCベース）
            # 冬時間Open 14:30, 夏時間Open 13:30
            # 簡易的に月で判定
            month = context.timestamp.month
            is_summer = 3 <= month <= 11
            
            target_hour = 13 if is_summer else 14
            target_minute = 30
            
            ts_hour = context.timestamp.hour
            ts_minute = context.timestamp.minute
            
            current_min = ts_hour * 60 + ts_minute
            target_min = target_hour * 60 + target_minute
            
            window = params.get("window_minutes", 30)
            
            return abs(current_min - target_min) <= window

    def get_default_params(self) -> Dict[str, Any]:
        """
        デフォルトパラメータ

        Returns:
            enabled=True, window_minutes=30
        """
        return {
            "enabled": True,
            "window_minutes": 30
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
            
        # 20%の確率でウィンドウサイズを変更
        if random.random() < 0.2:
            current = new_params.get("window_minutes", 30)
            delta = random.randint(-10, 10)
            new_params["window_minutes"] = max(10, min(60, current + delta))

        return new_params


# グローバルインスタンスを作成してレジストリに登録
us_market_open_filter = USMarketOpenFilter()
register_tool(us_market_open_filter)
