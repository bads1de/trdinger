import pytest
from unittest.mock import MagicMock, call
import pandas as pd
from app.services.auto_strategy.strategies.order_manager import OrderManager
from app.services.auto_strategy.config.constants import EntryType
from app.services.auto_strategy.positions.pending_order import PendingOrder

class TestOrderManagerEnhancement:
    @pytest.fixture
    def strategy(self):
        strategy = MagicMock()
        strategy.position = None
        strategy.base_timeframe = "1h"
        return strategy

    @pytest.fixture
    def simulator(self):
        return MagicMock()

    @pytest.fixture
    def manager(self, strategy, simulator):
        return OrderManager(strategy, simulator)

    @pytest.fixture
    def minute_data(self):
        return pd.DataFrame({"close": [100, 101, 102]}, index=pd.to_datetime(["2023-01-01 10:00", "2023-01-01 10:01", "2023-01-01 10:02"]))

    def test_check_pending_order_fills_empty(self, manager, minute_data):
        manager.pending_orders = []
        manager.check_pending_order_fills(minute_data, pd.Timestamp("2023-01-01 10:00"), 0)
        manager.lower_tf_simulator.get_minute_data_for_bar.assert_not_called()

    def test_check_pending_order_fills_existing_position(self, manager, minute_data, strategy):
        # ポジションがある場合
        strategy.position = MagicMock()
        manager.pending_orders = [MagicMock()]
        
        manager.check_pending_order_fills(minute_data, pd.Timestamp("2023-01-01 10:00"), 0)
        
        # 注文はクリアされるはず
        assert len(manager.pending_orders) == 0
        manager.lower_tf_simulator.get_minute_data_for_bar.assert_not_called()

    def test_check_pending_order_fills_no_minute_data(self, manager):
        manager.pending_orders = [MagicMock()]
        manager.check_pending_order_fills(None, pd.Timestamp("2023-01-01 10:00"), 0)
        manager.lower_tf_simulator.get_minute_data_for_bar.assert_not_called()

    def test_check_pending_order_fills_filled(self, manager, strategy, simulator, minute_data):
        # 注文設定
        order = MagicMock(spec=PendingOrder)
        order.is_long = True
        order.size = 1.0
        order.sl_price = 90
        order.tp_price = 110
        order.direction = 1
        manager.pending_orders = [order]
        
        # シミュレータ設定
        simulator.get_minute_data_for_bar.return_value = minute_data
        simulator.check_order_fill.return_value = (True, 101.0) # 約定, 価格
        
        # 実行
        manager.check_pending_order_fills(minute_data, pd.Timestamp("2023-01-01 10:00"), 0)
        
        # 検証
        simulator.get_minute_data_for_bar.assert_called_once()
        simulator.check_order_fill.assert_called_once_with(order, minute_data)
        
        # 約定処理
        strategy.buy.assert_called_once_with(size=1.0)
        assert strategy._entry_price == 101.0
        assert strategy._sl_price == 90
        assert strategy._tp_price == 110
        
        # 注文リストから削除されているか
        assert len(manager.pending_orders) == 0

    def test_check_pending_order_fills_not_filled(self, manager, simulator, minute_data):
        order = MagicMock(spec=PendingOrder)
        manager.pending_orders = [order]
        
        simulator.get_minute_data_for_bar.return_value = minute_data
        simulator.check_order_fill.return_value = (False, None)
        
        manager.check_pending_order_fills(minute_data, pd.Timestamp("2023-01-01 10:00"), 0)
        
        # 注文は残る
        assert len(manager.pending_orders) == 1

    def test_expire_pending_orders(self, manager):
        order1 = MagicMock()
        order1.is_expired.return_value = False
        
        order2 = MagicMock()
        order2.is_expired.return_value = True
        
        manager.pending_orders = [order1, order2]
        
        manager.expire_pending_orders(10)
        
        assert len(manager.pending_orders) == 1
        assert manager.pending_orders[0] == order1
        
        order1.is_expired.assert_called_with(10)
        order2.is_expired.assert_called_with(10)

    def test_create_pending_order(self, manager):
        entry_gene = MagicMock()
        entry_gene.entry_type = EntryType.LIMIT
        entry_gene.order_validity_bars = 5
        
        manager.create_pending_order(
            direction=1,
            size=0.5,
            entry_params={"limit": 100, "stop": None},
            sl_price=90,
            tp_price=110,
            entry_gene=entry_gene,
            current_bar_index=10
        )
        
        assert len(manager.pending_orders) == 1
        order = manager.pending_orders[0]
        assert order.order_type == EntryType.LIMIT
        assert order.limit_price == 100
        assert order.validity_bars == 5
        assert order.created_bar_index == 10

    def test_get_bar_duration(self, manager, strategy):
        strategy.base_timeframe = "15m"
        duration = manager._get_bar_duration()
        assert duration == pd.Timedelta(minutes=15)
        
        strategy.base_timeframe = "invalid"
        duration = manager._get_bar_duration()
        assert duration is None
