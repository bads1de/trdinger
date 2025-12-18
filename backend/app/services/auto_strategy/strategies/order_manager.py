"""
注文管理マネージャー

保留注文（PendingOrder）のライフサイクル管理、約定チェック、有効期限管理を担当します。
UniversalStrategyから注文管理の責務を分離するためのクラスです。
"""

import logging
from typing import List, Optional

import pandas as pd

from ..config.constants import EntryType
from ..genes.entry import EntryGene
from ..positions.lower_tf_simulator import LowerTimeframeSimulator
from ..positions.pending_order import PendingOrder

logger = logging.getLogger(__name__)


class OrderManager:
    """
    注文管理マネージャー

    保留注文の管理と執行ロジックをカプセル化します。
    """

    def __init__(self, strategy, lower_tf_simulator: LowerTimeframeSimulator):
        """
        初期化

        Args:
            strategy: UniversalStrategyのインスタンス（buy/sell実行用）
            lower_tf_simulator: 1分足シミュレーター
        """
        self.strategy = strategy
        self.lower_tf_simulator = lower_tf_simulator
        self.pending_orders: List[PendingOrder] = []

    def check_pending_order_fills(
        self, minute_data: pd.DataFrame, current_bar_time, current_bar_index: int
    ) -> None:
        """
        保留注文の約定をチェック

        Args:
            minute_data: 1分足データ
            current_bar_time: 現在のバーの開始時刻
            current_bar_index: 現在のバーインデックス
        """
        if not self.pending_orders or minute_data is None:
            return

        if self.strategy.position:
            # 既にポジションがある場合は保留注文をキャンセル
            self.pending_orders.clear()
            return

        bar_duration = self._get_bar_duration()
        if bar_duration is None:
            return

        # 期間: [OpenTime, OpenTime + Duration)
        bar_start = current_bar_time
        bar_end = current_bar_time + bar_duration

        # 該当期間の1分足を抽出
        minute_bars = self.lower_tf_simulator.get_minute_data_for_bar(
            minute_data, bar_start, bar_end
        )

        if minute_bars.empty:
            return

        filled_orders = []

        for order in self.pending_orders:
            filled, fill_price = self.lower_tf_simulator.check_order_fill(
                order, minute_bars
            )

            if filled and fill_price is not None:
                # 約定実行
                self._execute_filled_order(order, fill_price)
                filled_orders.append(order)
                break  # 1バーで1注文のみ約定

        # 約定した注文を削除
        for order in filled_orders:
            self.pending_orders.remove(order)

    def expire_pending_orders(self, current_bar_index: int) -> None:
        """
        期限切れの保留注文を削除

        Args:
            current_bar_index: 現在のバーインデックス
        """
        self.pending_orders = [
            order
            for order in self.pending_orders
            if not order.is_expired(current_bar_index)
        ]

    def create_pending_order(
        self,
        direction: float,
        size: float,
        entry_params: dict,
        sl_price: Optional[float],
        tp_price: Optional[float],
        entry_gene: EntryGene,
        current_bar_index: int,
    ) -> None:
        """
        保留注文を作成してリストに追加

        Args:
            direction: 取引方向
            size: ポジションサイズ
            entry_params: エントリーパラメータ
            sl_price: SL価格
            tp_price: TP価格
            entry_gene: エントリー遺伝子
            current_bar_index: 現在のバーインデックス
        """
        order = PendingOrder(
            order_type=entry_gene.entry_type,
            direction=direction,
            limit_price=entry_params.get("limit"),
            stop_price=entry_params.get("stop"),
            size=size,
            created_bar_index=current_bar_index,
            validity_bars=entry_gene.order_validity_bars,
            sl_price=sl_price,
            tp_price=tp_price,
        )
        self.pending_orders.append(order)

    def _execute_filled_order(self, order: PendingOrder, fill_price: float) -> None:
        """
        約定した注文を実行（Strategyに委譲）

        Args:
            order: 約定した注文
            fill_price: 約定価格
        """
        if order.is_long():
            self.strategy.buy(size=order.size)
        else:
            self.strategy.sell(size=order.size)

        # 戦略の内部状態を更新（コールバック的に処理）
        # UniversalStrategy側で公開されているメソッドや属性を操作する
        # ここでは直接属性を操作するが、setterメソッドがあればそちらが望ましい
        if hasattr(self.strategy, "_entry_price"):
            self.strategy._entry_price = fill_price
        if hasattr(self.strategy, "_sl_price"):
            self.strategy._sl_price = order.sl_price
        if hasattr(self.strategy, "_tp_price"):
            self.strategy._tp_price = order.tp_price
        if hasattr(self.strategy, "_position_direction"):
            self.strategy._position_direction = order.direction

    def _get_bar_duration(self) -> Optional[pd.Timedelta]:
        """現在のタイムフレームのバー期間を取得"""
        # UniversalStrategyからbase_timeframeを取得
        timeframe = getattr(self.strategy, "base_timeframe", "1h")
        
        timeframe_map = {
            "1m": pd.Timedelta(minutes=1),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1),
        }
        return timeframe_map.get(timeframe)
