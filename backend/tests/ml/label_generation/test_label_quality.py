import pytest
import pandas as pd
import numpy as np
from app.services.ml.label_generation.triple_barrier import TripleBarrier


class TestLabelQuality:
    """ラベルの品質（収益性）を検証するテスト"""

    @pytest.fixture
    def sample_market_data(self):
        """テスト用の市場データ（トレンドを含む）を生成"""
        # 1000本のデータ
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1H")

        # ランダムウォーク + トレンド
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)

        # 特定の区間に強い上昇トレンドを注入 (index 200-300)
        returns[200:300] += 0.02

        # 特定の区間に強い下降トレンドを注入 (index 600-700)
        returns[600:700] -= 0.02

        price = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "open": price,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "volume": np.random.randint(100, 1000, 1000),
            },
            index=dates,
        )

        return df

    def test_triple_barrier_profitability(self, sample_market_data):
        """Triple Barrierラベルが将来の利益と相関しているか検証"""
        tb = TripleBarrier(pt=1.0, sl=1.0, min_ret=0.005, num_threads=1)  # 0.5%

        close = sample_market_data["close"]

        # ボラティリティ計算 (簡易版)
        volatility = close.pct_change().rolling(window=20).std()
        volatility = volatility.fillna(0.005)  # デフォルト値

        # 全タイムスタンプをイベントとする
        t_events = close.index

        # 垂直バリア (24時間後)
        vertical_barrier = pd.Series(
            close.index + pd.Timedelta(hours=24), index=close.index
        )

        # イベント取得
        events = tb.get_events(
            close=close,
            t_events=t_events,
            pt_sl=[1.0, 1.0],
            target=volatility,
            min_ret=0.005,
            vertical_barrier_times=vertical_barrier,
        )

        # ラベル生成 (バイナリ: 1=Profit, 0=Loss/TimeOut)
        labels = tb.get_bins(events, close, binary_label=True)

        # 検証: ラベル1の平均リターンはプラスであるべき
        # get_binsの戻り値には 'ret' カラムが含まれている

        # ラベル1 (Trend/Profit) のリターン
        returns_1 = labels[labels["bin"] == 1]["ret"]

        # ラベル0 (No Trend/Loss) のリターン
        returns_0 = labels[labels["bin"] == 0]["ret"]

        print(f"\nLabel 1 Count: {len(returns_1)}")
        print(f"Label 0 Count: {len(returns_0)}")
        print(f"Label 1 Mean Return: {returns_1.mean():.4f}")
        print(f"Label 0 Mean Return: {returns_0.mean():.4f}")

        # アサーション
        # 1. ラベル1が一定数存在すること (トレンド注入区間があるため)
        assert len(returns_1) > 10, "トレンドラベルが少なすぎます"

        # 2. ラベル1の平均リターンがプラスであること
        assert (
            returns_1.mean() > 0
        ), "トレンドラベルの平均リターンがプラスではありません"

        # 3. ラベル1のリターンがラベル0より有意に高いこと
        assert (
            returns_1.mean() > returns_0.mean()
        ), "トレンドラベルが非トレンドラベルより優位ではありません"

    def test_triple_barrier_with_side(self, sample_market_data):
        """サイド指定ありの場合のTriple Barrier検証"""
        tb = TripleBarrier(pt=1.0, sl=1.0, min_ret=0.005)
        close = sample_market_data["close"]
        volatility = close.pct_change().rolling(window=20).std().fillna(0.005)

        # 常に「買い」を指示するサイドシグナル
        side = pd.Series(1, index=close.index)

        events = tb.get_events(
            close=close,
            t_events=close.index,
            pt_sl=[1.0, 1.0],
            target=volatility,
            min_ret=0.005,
            vertical_barrier_times=pd.Series(
                close.index + pd.Timedelta(hours=24), index=close.index
            ),
            side=side,
        )

        labels = tb.get_bins(events, close, binary_label=True)

        # 上昇トレンド区間 (200-300) ではラベル1が多くなるはず
        trend_section_labels = labels.iloc[200:280]  # 少し余裕を持たせる
        positive_ratio = (trend_section_labels["bin"] == 1).mean()

        print(f"\nTrend Section Positive Ratio: {positive_ratio:.2%}")

        assert (
            positive_ratio > 0.3
        ), "上昇トレンド区間で正解ラベルが十分に生成されていません"
