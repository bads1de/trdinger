import pytest
import pandas as pd
import numpy as np
from app.services.ml.label_generation.cusum_generator import CusumSignalGenerator


class TestCusumSignalGenerator:
    @pytest.fixture
    def sample_data(self):
        """
        テスト用のサンプルデータを生成
        """
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        # 初期価格100
        prices = [100.0]

        # 0-20: レンジ (変動なし)
        for _ in range(20):
            prices.append(prices[-1])

        # 21-40: 上昇トレンド (毎回 +1%)
        for _ in range(20):
            prices.append(prices[-1] * 1.01)

        # 41-60: レンジ
        for _ in range(20):
            prices.append(prices[-1])

        # 61-80: 下降トレンド (毎回 -1%)
        for _ in range(20):
            prices.append(prices[-1] * 0.99)

        # 81-99: レンジ
        for _ in range(19):
            prices.append(prices[-1])

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": 1000.0,
            },
            index=dates,
        )

        return df

    def test_get_events_uptrend(self, sample_data):
        """上昇トレンドでのイベント検出テスト"""
        generator = CusumSignalGenerator()

        # 閾値を0.02 (2%) に設定
        # 21-40の上昇局面で、累積が2%を超えるごとにイベントが出るはず
        events = generator.get_events(
            sample_data, threshold=0.02, volatility=None  # 固定閾値テストのためNone
        )

        # イベントが検出されていること
        assert len(events) > 0

        # 最初の上昇イベントは21行目以降にあるはず
        first_event = events[0]
        assert first_event > sample_data.index[20]

    def test_get_events_downtrend(self, sample_data):
        """下降トレンドでのイベント検出テスト"""
        generator = CusumSignalGenerator()

        events = generator.get_events(sample_data, threshold=0.02, volatility=None)

        # 下降局面（61-80）でもイベントが出るはず
        downtrend_start = sample_data.index[60]
        downtrend_events = [t for t in events if t > downtrend_start]

        assert len(downtrend_events) > 0

    def test_no_events_in_flat_range(self, sample_data):
        """レンジ相場でのイベント非検出テスト"""
        # 最初の20期間は完全にフラットなのでイベントは出ないはず
        flat_data = sample_data.iloc[:20]

        generator = CusumSignalGenerator()
        events = generator.get_events(flat_data, threshold=0.01)

        assert len(events) == 0

    def test_dynamic_threshold(self):
        """動的閾値（ボラティリティ）を使用したテスト"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        prices = np.random.normal(100, 1, 100).cumsum()  # ランダムウォーク

        df = pd.DataFrame({"close": prices}, index=dates)

        # ボラティリティを計算（単純な標準偏差）
        returns = df["close"].pct_change()
        volatility = returns.rolling(window=20).std()

        generator = CusumSignalGenerator()
        events = generator.get_events(
            df, threshold=None, volatility=volatility  # Noneの場合、volatilityを使う
        )

        # 何らかのイベントが検出されること（ランダムウォークなので確実ではないが、期間が長ければ出る）
        # ここではエラーにならず実行できることを確認
        assert isinstance(events, pd.DatetimeIndex)

    def test_get_daily_volatility(self, sample_data):
        """日次ボラティリティ計算のテスト"""
        generator = CusumSignalGenerator()

        vol = generator.get_daily_volatility(sample_data["close"], span=20)

        assert isinstance(vol, pd.Series)
        assert len(vol) == len(sample_data)
        # 最初のほうはNaNになる
        assert pd.isna(vol.iloc[0])




