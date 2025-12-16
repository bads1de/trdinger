"""
PendingOrder モデルのテスト
"""

import pytest
from app.services.auto_strategy.positions.pending_order import PendingOrder
from app.services.auto_strategy.config.constants import EntryType


class TestPendingOrder:
    """PendingOrder のテストクラス"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        order = PendingOrder(
            order_type=EntryType.LIMIT,
            direction=1.0,
        )
        assert order.order_type == EntryType.LIMIT
        assert order.direction == 1.0
        assert order.limit_price is None
        assert order.stop_price is None
        assert order.size == 0.01
        assert order.created_bar_index == 0
        assert order.validity_bars == 5
        assert order.sl_price is None
        assert order.tp_price is None
        assert order.stop_triggered is False

    def test_is_expired_within_validity(self):
        """有効期限内の判定"""
        order = PendingOrder(
            order_type=EntryType.LIMIT,
            direction=1.0,
            created_bar_index=10,
            validity_bars=5,
        )
        # 10 + 5 = 15 まで有効
        assert order.is_expired(10) is False
        assert order.is_expired(11) is False
        assert order.is_expired(14) is False

    def test_is_expired_at_boundary(self):
        """有効期限境界での判定"""
        order = PendingOrder(
            order_type=EntryType.LIMIT,
            direction=1.0,
            created_bar_index=10,
            validity_bars=5,
        )
        # 15バー目で期限切れ
        assert order.is_expired(15) is True
        assert order.is_expired(16) is True

    def test_is_expired_unlimited(self):
        """無制限有効期限の判定"""
        order = PendingOrder(
            order_type=EntryType.LIMIT,
            direction=1.0,
            created_bar_index=0,
            validity_bars=0,  # 無制限
        )
        assert order.is_expired(100) is False
        assert order.is_expired(10000) is False

    def test_is_limit_order(self):
        """指値注文判定"""
        limit_order = PendingOrder(order_type=EntryType.LIMIT, direction=1.0)
        stop_order = PendingOrder(order_type=EntryType.STOP, direction=1.0)
        stop_limit_order = PendingOrder(order_type=EntryType.STOP_LIMIT, direction=1.0)
        market_order = PendingOrder(order_type=EntryType.MARKET, direction=1.0)

        assert limit_order.is_limit_order() is True
        assert stop_order.is_limit_order() is False
        assert stop_limit_order.is_limit_order() is True
        assert market_order.is_limit_order() is False

    def test_is_stop_order(self):
        """逆指値注文判定"""
        limit_order = PendingOrder(order_type=EntryType.LIMIT, direction=1.0)
        stop_order = PendingOrder(order_type=EntryType.STOP, direction=1.0)
        stop_limit_order = PendingOrder(order_type=EntryType.STOP_LIMIT, direction=1.0)
        market_order = PendingOrder(order_type=EntryType.MARKET, direction=1.0)

        assert limit_order.is_stop_order() is False
        assert stop_order.is_stop_order() is True
        assert stop_limit_order.is_stop_order() is True
        assert market_order.is_stop_order() is False

    def test_is_long(self):
        """ロング判定"""
        long_order = PendingOrder(order_type=EntryType.LIMIT, direction=1.0)
        short_order = PendingOrder(order_type=EntryType.LIMIT, direction=-1.0)

        assert long_order.is_long() is True
        assert long_order.is_short() is False
        assert short_order.is_long() is False
        assert short_order.is_short() is True

    def test_full_order_creation(self):
        """完全な注文作成"""
        order = PendingOrder(
            order_type=EntryType.STOP_LIMIT,
            direction=-1.0,
            limit_price=99.5,
            stop_price=100.5,
            size=0.05,
            created_bar_index=100,
            validity_bars=10,
            sl_price=101.0,
            tp_price=98.0,
        )

        assert order.order_type == EntryType.STOP_LIMIT
        assert order.direction == -1.0
        assert order.limit_price == 99.5
        assert order.stop_price == 100.5
        assert order.size == 0.05
        assert order.created_bar_index == 100
        assert order.validity_bars == 10
        assert order.sl_price == 101.0
        assert order.tp_price == 98.0
        assert order.is_short() is True
        assert order.is_limit_order() is True
        assert order.is_stop_order() is True
