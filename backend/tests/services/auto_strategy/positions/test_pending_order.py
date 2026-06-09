"""
pending_order モジュールのユニットテスト
"""

import pytest

from app.services.auto_strategy.config.constants import EntryType
from app.services.auto_strategy.positions.pending_order import PendingOrder


class TestPendingOrder:
    def test_creation_defaults(self):
        order = PendingOrder(order_type=EntryType.MARKET, direction=1.0)
        assert order.order_type == EntryType.MARKET
        assert order.direction == 1.0
        assert order.size == 0.01
        assert order.validity_bars == 5
        assert order.stop_triggered is False

    def test_is_expired_false(self):
        order = PendingOrder(order_type=EntryType.MARKET, direction=1.0, created_bar_index=0, validity_bars=5)
        assert order.is_expired(3) is False

    def test_is_expired_true(self):
        order = PendingOrder(order_type=EntryType.MARKET, direction=1.0, created_bar_index=0, validity_bars=5)
        assert order.is_expired(5) is True
        assert order.is_expired(10) is True

    def test_is_expired_zero_validity_never_expires(self):
        order = PendingOrder(order_type=EntryType.MARKET, direction=1.0, validity_bars=0)
        assert order.is_expired(100) is False

    def test_is_expired_with_created_bar_index(self):
        order = PendingOrder(order_type=EntryType.MARKET, direction=1.0, created_bar_index=5, validity_bars=3)
        assert order.is_expired(7) is False
        assert order.is_expired(8) is True

    def test_is_limit_order(self):
        order = PendingOrder(order_type=EntryType.LIMIT, direction=1.0)
        assert order.is_limit_order is True

    def test_is_limit_order_stop_limit(self):
        order = PendingOrder(order_type=EntryType.STOP_LIMIT, direction=1.0)
        assert order.is_limit_order is True

    def test_is_not_limit_order_market(self):
        order = PendingOrder(order_type=EntryType.MARKET, direction=1.0)
        assert order.is_limit_order is False

    def test_is_stop_order(self):
        order = PendingOrder(order_type=EntryType.STOP, direction=1.0)
        assert order.is_stop_order is True

    def test_is_stop_order_stop_limit(self):
        order = PendingOrder(order_type=EntryType.STOP_LIMIT, direction=1.0)
        assert order.is_stop_order is True

    def test_is_not_stop_order_market(self):
        order = PendingOrder(order_type=EntryType.MARKET, direction=1.0)
        assert order.is_stop_order is False

    def test_is_long(self):
        order = PendingOrder(order_type=EntryType.MARKET, direction=1.0)
        assert order.is_long is True

    def test_is_short(self):
        order = PendingOrder(order_type=EntryType.MARKET, direction=-1.0)
        assert order.is_short is True

    def test_is_not_long_when_short(self):
        order = PendingOrder(order_type=EntryType.MARKET, direction=-1.0)
        assert order.is_long is False

    def test_is_not_short_when_long(self):
        order = PendingOrder(order_type=EntryType.MARKET, direction=1.0)
        assert order.is_short is False
