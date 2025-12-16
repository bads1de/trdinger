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

        Args:
            order: 保留注文
            minute_data: 該当期間の1分足OHLCデータ（時系列順）

        Returns:
            (filled, fill_price): 約定したかとその価格のタプル
        """
        if minute_data.empty:
            return False, None

        # 注文タイプに応じた約定判定
        if order.order_type == EntryType.LIMIT:
            return self._check_limit_fill(order, minute_data)
        elif order.order_type == EntryType.STOP:
            return self._check_stop_fill(order, minute_data)
        elif order.order_type == EntryType.STOP_LIMIT:
            return self._check_stop_limit_fill(order, minute_data)
        else:
            # MARKET は保留されないはず
            logger.warning(f"成行注文が保留リストに含まれています: {order}")
            return False, None

    def _check_limit_fill(
        self,
        order: PendingOrder,
        minute_data: pd.DataFrame,
    ) -> Tuple[bool, Optional[float]]:
        """
        指値注文の約定判定

        - Long: Low ≤ limit_price で約定
        - Short: High ≥ limit_price で約定
        """
        if order.limit_price is None:
            return False, None

        for idx in range(len(minute_data)):
            row = minute_data.iloc[idx]
            low = row.get("Low", row.get("low", float("inf")))
            high = row.get("High", row.get("high", float("-inf")))

            if order.is_long():
                # Long指値: 価格が下落して指値に到達
                if low <= order.limit_price:
                    return True, order.limit_price
            else:
                # Short指値: 価格が上昇して指値に到達
                if high >= order.limit_price:
                    return True, order.limit_price

        return False, None

    def _check_stop_fill(
        self,
        order: PendingOrder,
        minute_data: pd.DataFrame,
    ) -> Tuple[bool, Optional[float]]:
        """
        逆指値注文の約定判定

        - Long: High ≥ stop_price で約定（ブレイクアウト買い）
        - Short: Low ≤ stop_price で約定（ブレイクアウト売り）
        """
        if order.stop_price is None:
            return False, None

        for idx in range(len(minute_data)):
            row = minute_data.iloc[idx]
            low = row.get("Low", row.get("low", float("inf")))
            high = row.get("High", row.get("high", float("-inf")))

            if order.is_long():
                # Long逆指値: 価格が上昇してストップに到達
                if high >= order.stop_price:
                    return True, order.stop_price
            else:
                # Short逆指値: 価格が下落してストップに到達
                if low <= order.stop_price:
                    return True, order.stop_price

        return False, None

    def _check_stop_limit_fill(
        self,
        order: PendingOrder,
        minute_data: pd.DataFrame,
    ) -> Tuple[bool, Optional[float]]:
        """
        逆指値指値注文の約定判定

        1. まずストップ価格に到達（トリガー）
        2. その後、指値価格に到達で約定
        """
        if order.stop_price is None or order.limit_price is None:
            return False, None

        stop_triggered = order.stop_triggered

        for idx in range(len(minute_data)):
            row = minute_data.iloc[idx]
            low = row.get("Low", row.get("low", float("inf")))
            high = row.get("High", row.get("high", float("-inf")))

            if not stop_triggered:
                # ストップトリガー判定
                if order.is_long():
                    if high >= order.stop_price:
                        stop_triggered = True
                        order.stop_triggered = True
                else:
                    if low <= order.stop_price:
                        stop_triggered = True
                        order.stop_triggered = True

            if stop_triggered:
                # 指値約定判定
                if order.is_long():
                    # ストップ発動後、指値で買い
                    if low <= order.limit_price:
                        return True, order.limit_price
                else:
                    # ストップ発動後、指値で売り
                    if high >= order.limit_price:
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
