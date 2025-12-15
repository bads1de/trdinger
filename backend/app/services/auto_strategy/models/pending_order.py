"""
保留注文モデル

指値/逆指値注文の保留状態を管理するデータクラス
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config.enums import EntryType


@dataclass
class PendingOrder:
    """
    保留中のエントリー注文

    エントリーシグナル発生時に作成され、1分足シミュレーションで
    約定判定されるまで保持される。
    """

    # 注文タイプ
    order_type: EntryType

    # 取引方向: 1.0=Long, -1.0=Short
    direction: float

    # 指値価格（LIMIT, STOP_LIMIT で使用）
    limit_price: Optional[float] = None

    # 逆指値価格（STOP, STOP_LIMIT で使用）
    stop_price: Optional[float] = None

    # ポジションサイズ
    size: float = 0.01

    # 注文作成時のバーインデックス
    created_bar_index: int = 0

    # 注文有効期限（バー数）
    validity_bars: int = 5

    # 約定時に設定するSL価格
    sl_price: Optional[float] = None

    # 約定時に設定するTP価格
    tp_price: Optional[float] = None

    # STOP_LIMIT用: ストップ発動済みフラグ
    stop_triggered: bool = False

    def is_expired(self, current_bar_index: int) -> bool:
        """
        注文が期限切れかどうかを判定

        Args:
            current_bar_index: 現在のバーインデックス

        Returns:
            期限切れの場合 True
        """
        if self.validity_bars == 0:
            # 0 は無制限
            return False
        bars_elapsed = current_bar_index - self.created_bar_index
        return bars_elapsed >= self.validity_bars

    def is_limit_order(self) -> bool:
        """指値注文かどうか"""
        return self.order_type in (EntryType.LIMIT, EntryType.STOP_LIMIT)

    def is_stop_order(self) -> bool:
        """逆指値注文かどうか"""
        return self.order_type in (EntryType.STOP, EntryType.STOP_LIMIT)

    def is_long(self) -> bool:
        """ロング注文かどうか"""
        return self.direction > 0

    def is_short(self) -> bool:
        """ショート注文かどうか"""
        return self.direction < 0
