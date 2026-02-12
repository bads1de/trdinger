"""
米国ランチタイムフィルター

米国東部時間（EST/EDT）のランチタイム（12:00-13:00）における流動性低下と
不規則な値動きを回避するためのフィルターです。
"""

from typing import Any, Dict
import pandas as pd

from .base import BaseTool, ToolContext
from .registry import register_tool


class USLunchFilter(BaseTool):
    """
    米国ランチタイムフィルター

    米国株式市場のランチタイム（12:00 - 13:00 EST/EDT）をスキップします。
    この時間帯は「エアポケット」と呼ばれ、流動性が落ちて価格形成が不安定になる傾向があります。

    EST (冬時間): UTC-5 -> ランチは UTC 17:00 - 18:00
    EDT (夏時間): UTC-4 -> ランチは UTC 16:00 - 17:00
    """

    @property
    def name(self) -> str:
        return "us_lunch_filter"

    @property
    def description(self) -> str:
        return "米国ランチタイム（12:00-13:00 EST）の流動性低下を回避します"

    def _is_dst(self, timestamp: pd.Timestamp) -> bool:
        """
        米国夏時間（DST）判定

        原則: 3月の第2日曜日 午前2時から 11月の第1日曜日 午前2時まで
        """
        year = timestamp.year
        month = timestamp.month

        if month < 3 or month > 11:
            return False
        if month > 3 and month < 11:
            return True

        # 3月と11月の詳細判定
        # その月の第N日曜日を求める簡易ロジック
        # (ここでは厳密性より計算速度を優先し、近似的な判定を行う場合もあるが、
        #  今回は標準的なpd.Timestampの機能で判定を試みる)

        # タイムゾーン変換で判定するのが最も確実だが、
        # timestampがnaiveなUTCであることを前提に簡易計算する

        # 3月の第2日曜日
        march_start = pd.Timestamp(f"{year}-03-01")
        # 曜日 (0=Mon, 6=Sun)
        # 最初の土曜日までの日数 + 8 で第2日曜日の日付が出る
        # (ただし計算が複雑になるため、簡易的に pytz を使いたいところだが、
        #  標準ライブラリ依存を避けるため、pandasの機能を使う)

        try:
            # pandasのtz_localizeを使って判定するのがベスト
            # UTCとして解釈し、US/Easternに変換してオフセットを確認
            localized = timestamp.tz_localize("UTC").tz_convert("US/Eastern")
            return localized.dst().total_seconds() != 0
        except Exception:
            # タイムゾーン情報がない場合などのフォールバック
            # 簡易的に夏時間を判定（誤差許容）
            return False

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
            # timestampがnaiveな場合、UTCとみなして変換
            ts = context.timestamp
            if ts.tz is None:
                ts = ts.tz_localize("UTC")

            ny_time = ts.tz_convert("US/Eastern")

            # 12:00 〜 13:00 の間ならスキップ
            # 12時台 (12:00:00 〜 12:59:59)
            return ny_time.hour == 12

        except Exception:
            # 変換失敗時は安全側に倒してスキップしない、あるいはUTCで概算
            # UTCでの概算（冬時間17時、夏時間16時）
            hour = context.timestamp.hour
            # 簡易判定
            # 3-11月なら夏時間と仮定
            month = context.timestamp.month
            is_summer = 3 <= month <= 11

            if is_summer:
                return hour == 16
            else:
                return hour == 17

    def get_default_params(self) -> Dict[str, Any]:
        """
        デフォルトパラメータ

        Returns:
            enabled=True
        """
        return {"enabled": True}


# グローバルインスタンスを作成してレジストリに登録
us_lunch_filter = USLunchFilter()
register_tool(us_lunch_filter)
