"""
LowerTimeframeSimulator のテスト

1分足シミュレーションのユニットテスト
"""

import pandas as pd
import pytest
from app.services.auto_strategy.positions.pending_order import PendingOrder
from app.services.auto_strategy.config.constants import EntryType
from app.services.auto_strategy.positions.lower_tf_simulator import (
    LowerTimeframeSimulator,
)


@pytest.fixture
def simulator():
    """シミュレーターインスタンス"""
    return LowerTimeframeSimulator()


@pytest.fixture
def minute_data_up_trend():
    """上昇トレンドの1分足データ"""
    data = {
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "High": [101.0, 102.0, 103.0, 104.0, 105.0],
        "Low": [99.5, 100.5, 101.5, 102.5, 103.5],
        "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
    }
    index = pd.date_range("2024-01-01 10:00", periods=5, freq="1min")
    return pd.DataFrame(data, index=index)


@pytest.fixture
def minute_data_down_trend():
    """下降トレンドの1分足データ"""
    data = {
        "Open": [105.0, 104.0, 103.0, 102.0, 101.0],
        "High": [105.5, 104.5, 103.5, 102.5, 101.5],
        "Low": [104.0, 103.0, 102.0, 101.0, 100.0],
        "Close": [104.5, 103.5, 102.5, 101.5, 100.5],
    }
    index = pd.date_range("2024-01-01 10:00", periods=5, freq="1min")
    return pd.DataFrame(data, index=index)


class TestLimitOrderFill:
    """指値注文の約定テスト"""

    def test_long_limit_fill_success(self, simulator, minute_data_down_trend):
        """ロング指値約定: 価格下落で指値到達"""
        order = PendingOrder(
            order_type=EntryType.LIMIT,
            direction=1.0,
            limit_price=102.0,  # 下降トレンドで到達する価格
        )
        filled, price = simulator.check_order_fill(order, minute_data_down_trend)
        assert filled is True
        assert price == 102.0

    def test_long_limit_fill_failure(self, simulator, minute_data_up_trend):
        """ロング指値未約定: 価格上昇で指値に到達しない"""
        order = PendingOrder(
            order_type=EntryType.LIMIT,
            direction=1.0,
            limit_price=95.0,  # 上昇トレンドでは到達しない
        )
        filled, price = simulator.check_order_fill(order, minute_data_up_trend)
        assert filled is False
        assert price is None

    def test_short_limit_fill_success(self, simulator, minute_data_up_trend):
        """ショート指値約定: 価格上昇で指値到達"""
        order = PendingOrder(
            order_type=EntryType.LIMIT,
            direction=-1.0,
            limit_price=103.0,  # 上昇トレンドで到達する価格
        )
        filled, price = simulator.check_order_fill(order, minute_data_up_trend)
        assert filled is True
        assert price == 103.0

    def test_short_limit_fill_failure(self, simulator, minute_data_down_trend):
        """ショート指値未約定: 価格下落で指値に到達しない"""
        order = PendingOrder(
            order_type=EntryType.LIMIT,
            direction=-1.0,
            limit_price=110.0,  # 下降トレンドでは到達しない
        )
        filled, price = simulator.check_order_fill(order, minute_data_down_trend)
        assert filled is False
        assert price is None


class TestStopOrderFill:
    """逆指値注文の約定テスト"""

    def test_long_stop_fill_success(self, simulator, minute_data_up_trend):
        """ロング逆指値約定: ブレイクアウト買い"""
        order = PendingOrder(
            order_type=EntryType.STOP,
            direction=1.0,
            stop_price=103.0,  # 上昇トレンドで到達
        )
        filled, price = simulator.check_order_fill(order, minute_data_up_trend)
        assert filled is True
        assert price == 103.0

    def test_long_stop_fill_failure(self, simulator, minute_data_down_trend):
        """ロング逆指値未約定: 価格下落でストップに到達しない"""
        order = PendingOrder(
            order_type=EntryType.STOP,
            direction=1.0,
            stop_price=110.0,  # 下降トレンドでは到達しない
        )
        filled, price = simulator.check_order_fill(order, minute_data_down_trend)
        assert filled is False
        assert price is None

    def test_short_stop_fill_success(self, simulator, minute_data_down_trend):
        """ショート逆指値約定: ブレイクダウン売り"""
        order = PendingOrder(
            order_type=EntryType.STOP,
            direction=-1.0,
            stop_price=102.0,  # 下降トレンドで到達
        )
        filled, price = simulator.check_order_fill(order, minute_data_down_trend)
        assert filled is True
        assert price == 102.0

    def test_short_stop_fill_failure(self, simulator, minute_data_up_trend):
        """ショート逆指値未約定: 価格上昇でストップに到達しない"""
        order = PendingOrder(
            order_type=EntryType.STOP,
            direction=-1.0,
            stop_price=95.0,  # 上昇トレンドでは到達しない
        )
        filled, price = simulator.check_order_fill(order, minute_data_up_trend)
        assert filled is False
        assert price is None


class TestStopLimitOrderFill:
    """逆指値指値注文の約定テスト"""

    def test_long_stop_limit_fill_success(self, simulator):
        """ロング逆指値指値約定: ストップ発動後に指値で約定"""
        # 価格が103まで上昇後、101.5まで下落するデータ
        data = {
            "Open": [100.0, 102.0, 104.0, 103.0, 102.0],
            "High": [102.0, 104.0, 105.0, 104.0, 103.0],
            "Low": [99.0, 101.0, 103.0, 101.5, 101.0],
            "Close": [101.0, 103.0, 104.0, 102.0, 101.5],
        }
        index = pd.date_range("2024-01-01 10:00", periods=5, freq="1min")
        minute_data = pd.DataFrame(data, index=index)

        order = PendingOrder(
            order_type=EntryType.STOP_LIMIT,
            direction=1.0,
            stop_price=103.0,  # バー2でトリガー
            limit_price=102.0,  # バー3で約定
        )
        filled, price = simulator.check_order_fill(order, minute_data)
        assert filled is True
        assert price == 102.0
        assert order.stop_triggered is True

    def test_long_stop_limit_stop_triggered_no_limit(self, simulator):
        """ロング逆指値指値: ストップ発動したが指値に到達しない"""
        # 価格が103まで上昇後、さらに上昇
        data = {
            "Open": [100.0, 102.0, 104.0, 105.0, 106.0],
            "High": [102.0, 104.0, 106.0, 107.0, 108.0],
            "Low": [99.0, 101.0, 103.0, 104.0, 105.0],
            "Close": [101.0, 103.0, 105.0, 106.0, 107.0],
        }
        index = pd.date_range("2024-01-01 10:00", periods=5, freq="1min")
        minute_data = pd.DataFrame(data, index=index)

        order = PendingOrder(
            order_type=EntryType.STOP_LIMIT,
            direction=1.0,
            stop_price=103.0,  # バー2でトリガー
            limit_price=99.0,  # 到達しない
        )
        filled, price = simulator.check_order_fill(order, minute_data)
        assert filled is False
        assert price is None
        assert order.stop_triggered is True


class TestEmptyData:
    """空データのテスト"""

    def test_empty_minute_data(self, simulator):
        """空の1分足データでは約定しない"""
        order = PendingOrder(
            order_type=EntryType.LIMIT,
            direction=1.0,
            limit_price=100.0,
        )
        empty_df = pd.DataFrame()
        filled, price = simulator.check_order_fill(order, empty_df)
        assert filled is False
        assert price is None


class TestGetMinuteDataForBar:
    """バー期間抽出テスト"""

    def test_extract_minute_data(self, simulator, minute_data_up_trend):
        """1時間バーに該当する1分足を抽出"""
        bar_start = pd.Timestamp("2024-01-01 10:01")
        bar_end = pd.Timestamp("2024-01-01 10:04")

        result = simulator.get_minute_data_for_bar(
            minute_data_up_trend, bar_start, bar_end
        )
        assert len(result) == 3  # 10:01, 10:02, 10:03

    def test_extract_minute_data_empty_range(self, simulator, minute_data_up_trend):
        """範囲外の期間では空データ"""
        bar_start = pd.Timestamp("2024-01-01 11:00")
        bar_end = pd.Timestamp("2024-01-01 12:00")

        result = simulator.get_minute_data_for_bar(
            minute_data_up_trend, bar_start, bar_end
        )
        assert len(result) == 0
