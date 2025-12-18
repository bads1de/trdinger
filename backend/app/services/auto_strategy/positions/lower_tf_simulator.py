"""
下位足シミュレーター

1分足データを使用して保留注文の約定をシミュレートします。
"""

import logging
from typing import Optional, Tuple

import pandas as pd

from ..config.constants import EntryType
from ..positions.pending_order import PendingOrder

logger = logging.getLogger(__name__)


class LowerTimeframeSimulator:
    """
    下位足シミュレーター

    1分足データを使用して指値/逆指値注文の約定判定を行います。
    これにより、上位足のバー内での価格推移を考慮した精度の高い
    シミュレーションが可能になります。
    """

    def check_order_fill(
        self,
        order: PendingOrder,
        minute_data: pd.DataFrame,
    ) -> Tuple[bool, Optional[float]]:
        """
        1分足データを使用して注文の約定をチェック

        注文タイプ（指値、逆指値）に合わせて、高値・安値のベクトル
        を使用して、バー内での約定有無と約定価格を判定します。

        Args:
            order: 判定対象の未執行注文オブジェクト
            minute_data: 該当期間をカバーする1分足データ

        Returns:
            (約定したかどうか, 約定価格) のタプル
        """
        if minute_data.empty:
            return False, None

        # カラム名の正規化（Low/low, High/high）
        cols = {c.lower(): c for c in minute_data.columns}
        lows = minute_data[cols.get("low", "Low")]
        highs = minute_data[cols.get("high", "High")]

        if order.order_type == EntryType.LIMIT:
            if order.limit_price is None:
                return False, None
            mask = (
                (lows <= order.limit_price)
                if order.is_long
                else (highs >= order.limit_price)
            )
            if mask.any():
                return True, order.limit_price

        elif order.order_type == EntryType.STOP:
            if order.stop_price is None:
                return False, None
            mask = (
                (highs >= order.stop_price)
                if order.is_long
                else (lows <= order.stop_price)
            )
            if mask.any():
                return True, order.stop_price

        elif order.order_type == EntryType.STOP_LIMIT:
            return self._check_stop_limit_fill_vectorized(order, lows, highs)

        return False, None

    def _check_stop_limit_fill_vectorized(
        self, order: PendingOrder, lows: pd.Series, highs: pd.Series
    ) -> Tuple[bool, Optional[float]]:
        """逆指値指値注文の約定判定（ベクトル化）"""
        if order.stop_price is None or order.limit_price is None:
            return False, None

        # 1. ストップ条件の判定
        if not order.stop_triggered:
            stop_mask = (
                (highs >= order.stop_price)
                if order.is_long
                else (lows <= order.stop_price)
            )
            if stop_mask.any():
                order.stop_triggered = True
                # トリガーした以降のデータのみを対象に指値判定を行う必要がある
                trigger_idx = stop_mask.idxmax()
                lows = lows.loc[trigger_idx:]
                highs = highs.loc[trigger_idx:]
            else:
                return False, None

        # 2. 指値条件の判定
        limit_mask = (
            (lows <= order.limit_price)
            if order.is_long
            else (highs >= order.limit_price)
        )
        if limit_mask.any():
            return True, order.limit_price

        return False, None

    def get_minute_data_for_bar(
        self,
        minute_data: pd.DataFrame,
        bar_start: pd.Timestamp,
        bar_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        指定されたバー期間の1分足データを抽出

        Args:
            minute_data: 全期間の1分足データ
            bar_start: バー開始時刻
            bar_end: バー終了時刻

        Returns:
            該当期間の1分足データ
        """
        if minute_data.empty:
            return minute_data

        # インデックスがDatetimeIndexの場合
        if isinstance(minute_data.index, pd.DatetimeIndex):
            mask = (minute_data.index >= bar_start) & (minute_data.index < bar_end)
            return minute_data.loc[mask]

        return minute_data
