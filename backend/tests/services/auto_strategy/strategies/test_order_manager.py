"""
OrderManagerのユニットテスト
"""

import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd

from app.services.auto_strategy.strategies.order_manager import OrderManager
from app.services.auto_strategy.positions.pending_order import PendingOrder
from app.services.auto_strategy.config.constants import EntryType
from app.services.auto_strategy.genes.entry import EntryGene


class TestOrderManager:
    """OrderManagerのテストクラス"""

    @pytest.fixture
    def mock_strategy(self):
        strategy = MagicMock()
        strategy.position = None
        strategy.base_timeframe = "1h"
        return strategy

    @pytest.fixture
    def mock_simulator(self):
        return Mock()

    @pytest.fixture
    def order_manager(self, mock_strategy, mock_simulator):
        return OrderManager(mock_strategy, mock_simulator)

    def test_create_pending_order(self, order_manager):
        """保留注文の作成テスト"""
        entry_gene = EntryGene(entry_type=EntryType.LIMIT, order_validity_bars=5)
        
        order_manager.create_pending_order(
            direction=1.0,
            size=0.1,
            entry_params={"limit": 99.0},
            sl_price=95.0,
            tp_price=110.0,
            entry_gene=entry_gene,
            current_bar_index=100
        )
        
        assert len(order_manager.pending_orders) == 1
        order = order_manager.pending_orders[0]
        assert order.limit_price == 99.0
        assert order.direction == 1.0
        assert order.validity_bars == 5

    def test_expire_pending_orders(self, order_manager):
        """期限切れ注文の削除テスト"""
        # 有効期限 5 バーの注文 (作成時 100)
        order_manager.pending_orders.append(PendingOrder(
            order_type=EntryType.LIMIT, direction=1.0, created_bar_index=100, validity_bars=5
        ))
        # 有効期限なしの注文
        order_manager.pending_orders.append(PendingOrder(
            order_type=EntryType.LIMIT, direction=1.0, created_bar_index=100, validity_bars=0
        ))
        
        # 104バー時点：まだ期限内
        order_manager.expire_pending_orders(104)
        assert len(order_manager.pending_orders) == 2
        
        # 105バー時点：最初の注文が期限切れ
        order_manager.expire_pending_orders(105)
        assert len(order_manager.pending_orders) == 1
        assert order_manager.pending_orders[0].validity_bars == 0

    def test_check_pending_order_fills_cancels_on_position(self, order_manager, mock_strategy):
        """ポジションがある場合に保留注文をキャンセルするかテスト"""
        order_manager.pending_orders.append(Mock(spec=PendingOrder))
        
        # ポジションあり状態にする
        mock_strategy.position = Mock()
        
        order_manager.check_pending_order_fills(pd.DataFrame(), pd.Timestamp.now(), 100)
        
        # 注文がクリアされていること
        assert len(order_manager.pending_orders) == 0

    def test_execute_filled_order_updates_strategy(self, order_manager, mock_strategy):
        """約定時に戦略の内部状態が更新されるかテスト"""
        order = PendingOrder(
            order_type=EntryType.LIMIT,
            direction=1.0,
            size=0.5,
            sl_price=90.0,
            tp_price=120.0
        )
        
        # 属性が存在することをシミュレート
        mock_strategy._entry_price = 0
        mock_strategy._sl_price = 0
        
        order_manager._execute_filled_order(order, fill_price=100.0)
        
        # Strategyのbuyが呼ばれたか
        mock_strategy.buy.assert_called_once_with(size=0.5)
        # 内部状態が更新されたか
        assert mock_strategy._entry_price == 100.0
        assert mock_strategy._sl_price == 90.0
        assert mock_strategy._tp_price == 120.0
        assert mock_strategy._position_direction == 1.0

    def test_bar_duration_caching(self, mock_strategy, mock_simulator):
        """バー期間のキャッシュ機能テスト"""
        # OrderManagerの初期化時に_get_bar_durationが呼ばれる
        # 呼び出し回数をカウントするためにラップする
        
        # まず通常の初期化
        manager = OrderManager(mock_strategy, mock_simulator)
        assert manager.bar_duration == pd.Timedelta(hours=1)
        
        # _get_bar_duration をスパイ
        manager._get_bar_duration = Mock(wraps=manager._get_bar_duration)
        
        # check_pending_order_fills を呼び出す
        # minute_data が None でない場合のみ duration チェックまで進む
        minute_data = pd.DataFrame()
        manager.pending_orders.append(Mock()) # 注文あり
        
        manager.check_pending_order_fills(minute_data, pd.Timestamp.now(), 100)
        
        # _get_bar_duration は呼ばれていないはず（キャッシュ使用）
        manager._get_bar_duration.assert_not_called()
