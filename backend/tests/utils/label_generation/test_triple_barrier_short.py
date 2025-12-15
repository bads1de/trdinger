import pandas as pd
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.services.ml.label_generation.triple_barrier import TripleBarrier


class TestTripleBarrierShort:
    """ショートポジションのトリプルバリアテスト"""

    def setup_method(self):
        # 1時間足データ
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        self.close = pd.Series(100.0, index=dates)
        self.volatility = pd.Series(0.01, index=dates)  # 1% ボラティリティ

    def test_short_profit_taking(self):
        """ショートポジションの利益確定（価格下落）テスト"""
        # インデックス5で価格が3%下落
        self.close.iloc[0] = 100.0
        self.close.iloc[1:5] = 100.0
        self.close.iloc[5] = 97.0

        tb = TripleBarrier(num_threads=1)

        vertical_barriers = pd.Series(
            self.close.index + pd.Timedelta(hours=10), index=self.close.index
        )

        # Side = -1 (ショート)
        # 2%の下落で利確したい
        side = pd.Series(-1, index=self.close.index)

        events = tb.get_events(
            close=self.close,
            t_events=self.close.index[:10],
            pt_sl=[2.0, 2.0],
            target=self.volatility,
            min_ret=0.001,
            vertical_barrier_times=vertical_barriers,
            side=side,
        )

        labels = tb.get_bins(events, self.close)

        # ショートポジション: 3%の価格下落 > ターゲット1% * 2.0 = 2%下落
        # したがって利確（PT）にヒットするはず
        # PTヒット -> bin = 1.0 (利益)

        assert labels.loc[self.close.index[0], "bin"] == 1.0
        # イベントサイドの確認
        assert events.loc[self.close.index[0], "side"] == "pt"

    def test_short_stop_loss(self):
        """ショートポジションのストップロス（価格上昇）テスト"""
        # インデックス5で価格が3%上昇
        self.close.iloc[0] = 100.0
        self.close.iloc[1:5] = 100.0
        self.close.iloc[5] = 103.0

        tb = TripleBarrier(num_threads=1)

        vertical_barriers = pd.Series(
            self.close.index + pd.Timedelta(hours=10), index=self.close.index
        )

        side = pd.Series(-1, index=self.close.index)

        events = tb.get_events(
            close=self.close,
            t_events=self.close.index[:10],
            pt_sl=[2.0, 2.0],
            target=self.volatility,
            min_ret=0.001,
            vertical_barrier_times=vertical_barriers,
            side=side,
        )

        labels = tb.get_bins(events, self.close)

        # ショートポジション: 3%の価格上昇 > ターゲット1% * 2.0 = 2%
        # SLにヒットするはず
        # SLヒット -> bin = -1.0 (損失)

        assert labels.loc[self.close.index[0], "bin"] == -1.0
        assert events.loc[self.close.index[0], "side"] == "sl"

