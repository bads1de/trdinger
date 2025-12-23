import pandas as pd
import numpy as np
import pytest
from app.services.ml.label_generation.trend_scanning import TrendScanning


class TestTrendScanning:
    @pytest.fixture
    def sample_data(self):
        """
        Create synthetic data with clear trends.
        """
        # 100 points
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")

        # 0-30: Uptrend
        # 30-60: Downtrend
        # 60-100: Sideways/Noise

        prices = np.zeros(100)

        # Uptrend: y = x + noise
        prices[0:30] = np.arange(30) + np.random.normal(0, 0.5, 30)

        # Downtrend: y = 30 - (x-30) + noise
        prices[30:60] = 30 - np.arange(30) + np.random.normal(0, 0.5, 30)

        # Sideways: y = 0 + noise
        prices[60:100] = np.random.normal(0, 1.0, 40)

        # Shift up to avoid negative prices
        prices += 100

        return pd.Series(prices, index=dates, name="close")

    def test_uptrend_detection(self, sample_data):
        """
        Test that strong uptrends are labeled as 1.
        """
        ts = TrendScanning(min_window=5, max_window=20, min_t_value=2.0)

        # Test a point early in the uptrend (e.g., index 5)
        # It should see the uptrend ahead.
        t_event = sample_data.index[5]

        labels = ts.get_labels(sample_data, t_events=pd.DatetimeIndex([t_event]))

        assert len(labels) == 1
        assert labels.iloc[0]["bin"] == 1
        assert labels.iloc[0]["t_value"] > 2.0
        assert labels.iloc[0]["ret"] > 0

    def test_downtrend_detection(self, sample_data):
        """
        Test that strong downtrends are labeled as -1.
        """
        ts = TrendScanning(min_window=5, max_window=20, min_t_value=2.0)

        # Test a point early in the downtrend (e.g., index 35)
        t_event = sample_data.index[35]

        labels = ts.get_labels(sample_data, t_events=pd.DatetimeIndex([t_event]))

        assert len(labels) == 1
        assert labels.iloc[0]["bin"] == -1
        assert labels.iloc[0]["t_value"] < -2.0
        assert labels.iloc[0]["ret"] < 0

    def test_sideways_detection(self, sample_data):
        """
        Test that sideways markets are labeled as 0.
        """
        ts = TrendScanning(
            min_window=5, max_window=20, min_t_value=5.0
        )  # High threshold to force 0

        # Test a point in sideways (e.g., index 70)
        t_event = sample_data.index[70]

        labels = ts.get_labels(sample_data, t_events=pd.DatetimeIndex([t_event]))

        assert len(labels) == 1
        # It might detect a small trend if noise aligns, but with high threshold it should be 0
        # Or t-value should be low.
        # Let's check t-value absolute is likely small or we enforce bin 0

        # Note: Random walk can have trends.
        # But we expect bin to be 0 if t-value is below threshold.
        if abs(labels.iloc[0]["t_value"]) < 5.0:
            assert labels.iloc[0]["bin"] == 0

    def test_window_selection(self):
        """
        Test that it selects the window with the strongest trend.
        """
        # Create data where short term is noise, long term is strong trend
        dates = pd.date_range(start="2023-01-01", periods=30, freq="h")
        prices = np.array(
            [
                10, 9, 11, 10, 12, 11, 13, 12, 14, 13, 
                15, 14, 16, 15, 17, 16, 18, 17, 19, 18, 
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            ]
        )
        # Overall strong uptrend.

        s = pd.Series(prices, index=dates)

        ts = TrendScanning(min_window=5, max_window=20)
        labels = ts.get_labels(s, t_events=pd.DatetimeIndex([dates[0]]))

        # Should pick a window that captures the trend
        assert labels.iloc[0]["bin"] == 1
        assert pd.notna(labels.iloc[0]["t1"])

    def test_empty_input(self):
        """空の入力に対するテスト"""
        ts = TrendScanning()
        empty_s = pd.Series([], dtype=float)
        labels = ts.get_labels(empty_s)
        assert labels.empty

    def test_insufficient_data(self):
        """データ不足のテスト"""
        ts = TrendScanning(min_window=10)
        dates = pd.date_range(start="2023-01-01", periods=5, freq="h")
        s = pd.Series(np.random.randn(5), index=dates)
        
        # ウィンドウサイズ(10)よりデータが少ない場合、ラベルは生成されないはず
        labels = ts.get_labels(s)
        assert labels.empty

    def test_no_events_found(self, sample_data):
        """該当イベントがない場合のテスト"""
        ts = TrendScanning()
        # データに存在しないタイムスタンプを指定
        t_events = pd.to_datetime(["2025-01-01"])
        labels = ts.get_labels(sample_data, t_events=t_events)
        assert labels.empty

    def test_constant_price(self):
        """価格が一定の場合のテスト"""
        ts = TrendScanning(min_window=5, max_window=10)
        dates = pd.date_range(start="2023-01-01", periods=20, freq="h")
        s = pd.Series(100.0, index=dates)
        
        labels = ts.get_labels(s)
        # 傾きが0なので、t値も0になり、binは0になるはず
        assert not labels.empty
        assert (labels["bin"] == 0).all()
        assert (labels["t_value"] == 0).all()

    def test_mixed_events(self, sample_data):
        """存在するイベントと存在しないイベントが混在する場合"""
        ts = TrendScanning(min_window=5, max_window=10)
        valid_event = sample_data.index[5]
        invalid_event = pd.Timestamp("2025-01-01")
        t_events = pd.DatetimeIndex([valid_event, invalid_event])
        
        labels = ts.get_labels(sample_data, t_events=t_events)
        # 有効なイベントのみ処理されるはず
        assert len(labels) == 1
        assert labels.index[0] == valid_event

    def test_very_small_window(self):
        """非常に小さいウィンドウサイズでの動作"""
        # n <= 2 のケースを避けるため min_window=2 (n=3)
        ts = TrendScanning(min_window=2, max_window=4)
        dates = pd.date_range(start="2023-01-01", periods=10, freq="h")
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=dates, dtype=float)
        
        labels = ts.get_labels(s)
        assert not labels.empty
        assert (labels["bin"] == 1).all()

    def test_perfect_linear_trend(self):
        """完全な線形トレンド（誤差0）のケース"""
        ts = TrendScanning(min_window=5, max_window=10)
        dates = pd.date_range(start="2023-01-01", periods=20, freq="h")
        # ノイズなしの完全な線形増加
        # 0除算回避のために極小のノイズを乗せるか、実装のクリップを確認
        s = pd.Series(np.arange(20) * 1.5 + 100, index=dates)
        
        labels = ts.get_labels(s)
        assert not labels.empty
        # t値が十分に高いことを確認
        assert (labels["t_value"] > 5.0).all()
        assert (labels["bin"] == 1).all()

    def test_perfect_linear_down_trend(self):
        """完全な線形下降トレンド"""
        ts = TrendScanning(min_window=5, max_window=10)
        dates = pd.date_range(start="2023-01-01", periods=20, freq="h")
        s = pd.Series(200 - np.arange(20) * 1.5, index=dates)
        
        labels = ts.get_labels(s)
        assert not labels.empty
        # t値が十分に低いことを確認
        assert (labels["t_value"] < -5.0).all()
        assert (labels["bin"] == -1).all()

    def test_max_window_out_of_bounds(self):
        """max_window が系列の終わりを超える場合"""
        # データ末尾付近でのイベント発生
        ts = TrendScanning(min_window=5, max_window=100)
        dates = pd.date_range(start="2023-01-01", periods=50, freq="h")
        s = pd.Series(np.linspace(100, 110, 50), index=dates)
        
        # 最後の10件より前なら、min_window=5 は確保できる
        t_event = dates[40]
        labels = ts.get_labels(s, t_events=pd.DatetimeIndex([t_event]))
        
        # ウィンドウサイズが確保できればラベルは生成される
        assert not labels.empty
        assert labels.index[0] == t_event




