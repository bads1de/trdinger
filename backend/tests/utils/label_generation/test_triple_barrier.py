import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.services.ml.label_generation.triple_barrier import TripleBarrier


class TestTripleBarrier:
    """トリプルバリアラベル生成のテストクラス"""

    def setup_method(self):
        # 1時間ごとのデータ
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        self.close = pd.Series(100.0, index=dates)  # 初期値100
        # 変動を加える (ランダムウォーク)
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100)  # 1%の変動
        self.close = self.close * (1 + returns).cumprod()

        self.volatility = pd.Series(0.01, index=dates)  # ボラティリティ 1% (一定と仮定)

    def test_profit_taking(self):
        """利確バリアにヒットするケース"""
        # テスト用にデータを加工: index[0]から見て、index[5]で確実に上がるようにする
        # index[0] = 100.0 と仮定
        self.close.iloc[0] = 100.0
        self.close.iloc[1:5] = 100.0  # 途中は動かない
        self.close.iloc[5] = 103.0  # 5時間後に +3%

        tb = TripleBarrier(num_threads=1)

        # 垂直バリアは10時間後
        vertical_barriers = pd.Series(
            self.close.index + pd.Timedelta(hours=10), index=self.close.index
        )

        # pt=2.0 => 2.0 * vol(1%) = 2% 上昇で利確
        events = tb.get_events(
            close=self.close,
            t_events=self.close.index[:10],  # 最初の10個だけテスト
            pt_sl=[2.0, 2.0],
            target=self.volatility,
            min_ret=0.001,
            vertical_barrier_times=vertical_barriers,
        )

        # ラベル生成
        labels = tb.get_bins(events, self.close)

        # index[0] は +3% (index[5]) で利確ライン(+2%)を超えるはず
        assert labels.loc[self.close.index[0], "bin"] == 1.0

    def test_stop_loss(self):
        """損切りバリアにヒットするケース"""
        self.close.iloc[0] = 100.0
        self.close.iloc[1:5] = 100.0
        self.close.iloc[5] = 97.0  # 5時間後に -3%

        tb = TripleBarrier(num_threads=1)

        vertical_barriers = pd.Series(
            self.close.index + pd.Timedelta(hours=10), index=self.close.index
        )

        events = tb.get_events(
            close=self.close,
            t_events=self.close.index[:10],
            pt_sl=[2.0, 2.0],  # 2% 下落で損切り
            target=self.volatility,
            min_ret=0.001,
            vertical_barrier_times=vertical_barriers,
        )

        labels = tb.get_bins(events, self.close)

        # index[0] は -3% で損切りライン(-2%)に触れる -> -1
        assert labels.loc[self.close.index[0], "bin"] == -1.0

    def test_time_barrier(self):
        """時間切れのケース"""
        self.close.iloc[0] = 100.0
        self.close.iloc[1:10] = 100.5  # +0.5% (バリア 2% に届かない)

        tb = TripleBarrier(num_threads=1)

        # 垂直バリアを2時間後に設定
        vertical_barriers = pd.Series(
            self.close.index + pd.Timedelta(hours=2), index=self.close.index
        )

        events = tb.get_events(
            close=self.close,
            t_events=self.close.index[:10],
            pt_sl=[2.0, 2.0],
            target=self.volatility,
            min_ret=0.001,
            vertical_barrier_times=vertical_barriers,
        )

        labels = tb.get_bins(events, self.close)

        # バリアに触れずに時間切れ -> 0
        assert labels.loc[self.close.index[0], "bin"] == 0.0

    def test_events_match_manual_first_touch_logic(self):
        """Numba 実装が素朴な first-touch 判定と一致することを確認"""
        np.random.seed(42)
        n = 200
        index = pd.date_range(start="2024-01-01", periods=n, freq="1h")
        close_vals = np.concatenate(
            [
                np.linspace(100, 110, 50),
                np.linspace(110, 100, 50),
                100 + np.random.randn(100).cumsum(),
            ]
        )
        close = pd.Series(close_vals, index=index)
        target = pd.Series(0.01, index=index)

        tb = TripleBarrier(pt=1.0, sl=1.0, min_ret=0.0001)

        events_long = tb.get_events(
            close,
            index,
            [1.0, 1.0],
            target,
            0.0001,
            side=pd.Series(1, index=index),
        )
        events_short = tb.get_events(
            close,
            index,
            [1.0, 1.0],
            target,
            0.0001,
            side=pd.Series(-1, index=index),
        )

        assert not events_long.empty
        assert not events_short.empty

        def verify_point(events_df, idx_check, side_val):
            t0 = index[idx_check]
            row = events_df.loc[t0]

            numba_t1 = row["t1"]
            numba_side = row["side"]

            if pd.isna(numba_t1):
                p0 = close.iloc[idx_check]
                upper = p0 * (1 + 0.01)
                lower = p0 * (1 - 0.01)

                future_prices = close.iloc[idx_check + 1 :]
                if side_val == 1:
                    hit_pt = future_prices[future_prices >= upper]
                    hit_sl = future_prices[future_prices <= lower]
                else:
                    hit_pt = future_prices[future_prices <= lower]
                    hit_sl = future_prices[future_prices >= upper]

                assert (
                    hit_pt.empty and hit_sl.empty
                ), f"Numba missed a hit at {t0}. Side={side_val}"
                return

            p0 = close.iloc[idx_check]
            trgt = 0.01
            sub_period = close.loc[t0:numba_t1][1:]
            rets = sub_period / p0 - 1

            if side_val == 1:
                first_pt = rets[rets >= trgt].index.min()
                first_sl = rets[rets <= -trgt].index.min()
            else:
                first_pt = rets[rets <= -trgt].index.min()
                first_sl = rets[rets >= trgt].index.min()

            valid_events = [ts for ts in [first_pt, first_sl] if pd.notna(ts)]
            if not valid_events:
                raise AssertionError(
                    f"Python logic found no event, but Numba found {numba_t1}"
                )

            first_event = min(valid_events)
            assert (
                first_event == numba_t1
            ), f"Timing mismatch at {t0}. Numba={numba_t1}, Python={first_event}"

            ret_at_hit = close.loc[first_event] / p0 - 1
            if side_val == 1:
                expected_type = "pt" if ret_at_hit > 0 else "sl"
            else:
                expected_type = "pt" if ret_at_hit < 0 else "sl"

            assert numba_side == expected_type, (
                f"Side mismatch at {t0}. Numba={numba_side}, "
                f"Python={expected_type}, Ret={ret_at_hit}"
            )

        for i in np.random.choice(range(len(index) - 1), 10, replace=False):
            verify_point(events_long, i, 1)
            verify_point(events_short, i, -1)

    def test_triple_barrier_performance(self):
        """Triple Barrier のパフォーマンスを簡易確認する"""
        np.random.seed(42)
        n = 10000
        index = pd.date_range(start="2020-01-01", periods=n, freq="1h")
        close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.01) * 100, index=index)
        target = pd.Series(0.01, index=index)

        tb = TripleBarrier(pt=1.0, sl=1.0, min_ret=0.001)

        # JIT warmup
        _ = tb.get_events(
            close=close.iloc[:100],
            t_events=index[:100],
            pt_sl=[1.0, 1.0],
            target=target.iloc[:100],
            min_ret=0.001,
        )

        start_time = time.time()
        events = tb.get_events(
            close=close,
            t_events=index,
            pt_sl=[1.0, 1.0],
            target=target,
            min_ret=0.001,
        )
        duration = time.time() - start_time

        print(f"\nTriple Barrier (n={len(close)}): {duration:.4f} seconds")
        assert not events.empty
