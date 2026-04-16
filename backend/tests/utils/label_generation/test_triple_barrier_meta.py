import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.services.ml.label_generation.triple_barrier import TripleBarrier


class TestTripleBarrierMetaLabeling:
    """メタラベリングのためのトリプルバリアテスト"""

    def setup_method(self):
        # 1時間ごとのデータ
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        self.close = pd.Series(100.0, index=dates)
        self.volatility = pd.Series(0.01, index=dates)  # 1%

    def test_meta_labeling_binary_output(self):
        """メタラベリング(トレンド=1, その他=0)の動作テスト"""
        tb = TripleBarrier()

        # シナリオ1: 成功 (利確バリアヒット) -> 1
        # シナリオ2: 失敗 (損切りバリアヒット) -> 0
        # シナリオ3: 失敗 (時間切れ) -> 0

        # closeデータを構築
        # 0: スタート
        # 1-4: レンジ
        # 5: 成功ケース (index[0]からの判定) -> +3% (PT=2%)
        # 10: スタート2
        # 15: 失敗ケース (index[10]からの判定) -> -3% (SL=2%)
        # 20: スタート3
        # 21-30: 時間切れケース (index[20]からの判定) -> +0.5% (動かず)

        self.close.iloc[0] = 100.0
        self.close.iloc[5] = 103.0

        self.close.iloc[10] = 100.0
        self.close.iloc[15] = 97.0

        self.close.iloc[20] = 100.0
        self.close.iloc[30] = 100.5

        # 垂直バリア (5時間後)
        # index[0] -> limit index[5] (ちょうどヒット)
        # index[10] -> limit index[15]
        # index[20] -> limit index[25] (25番目の価格は100.0)
        vertical_barriers = pd.Series(
            cast(pd.DatetimeIndex, self.close.index) + pd.Timedelta(hours=6),
            index=self.close.index,
        )

        # ターゲットイベント
        t_events = pd.DatetimeIndex(
            [
                self.close.index[0],  # 成功予定
                self.close.index[10],  # 損切り予定
                self.close.index[20],  # 時間切れ予定
            ]
        )

        events = tb.get_events(
            close=self.close,
            t_events=t_events,
            pt_sl=[2.0, 2.0],
            target=self.volatility,
            min_ret=0.001,
            vertical_barrier_times=vertical_barriers,
        )

        # 新しい引数 `binary_label=True` を使用 (まだ実装していないので赤になるはず)
        # または get_bins_meta_labeling という新メソッド
        try:
            labels = tb.get_bins(events, self.close, binary_label=True)
        except TypeError:
            pytest.fail(
                "get_bins メソッドはまだ 'binary_label' パラメータをサポートしていません"
            )

        # 検証
        # ケース1: 利確 -> 1
        assert labels.loc[self.close.index[0], "bin"] == 1.0

        # ケース2: 損切り -> 0 (通常は-1だが、binary=Trueなら0)
        assert labels.loc[self.close.index[10], "bin"] == 0.0

        # ケース3: 時間切れ -> 0
        assert labels.loc[self.close.index[20], "bin"] == 0.0

    def test_binary_labels_correlate_with_future_profitability(self):
        """binary_label=True の出力がトレンド方向の収益性と整合することを確認"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")

        returns = np.random.normal(0, 0.01, 1000)
        returns[200:300] += 0.02
        returns[600:700] -= 0.02
        price = 100 * np.exp(np.cumsum(returns))

        close = pd.Series(price, index=dates)
        volatility = close.pct_change().rolling(window=20).std().fillna(0.005)

        tb = TripleBarrier(pt=1.0, sl=1.0, min_ret=0.005, num_threads=1)
        vertical_barrier = pd.Series(
            cast(pd.DatetimeIndex, close.index) + pd.Timedelta(hours=24),
            index=close.index,
        )

        events = tb.get_events(
            close=close,
            t_events=cast(pd.DatetimeIndex, close.index),
            pt_sl=[1.0, 1.0],
            target=volatility,
            min_ret=0.005,
            vertical_barrier_times=vertical_barrier,
        )

        labels = tb.get_bins(events, close, binary_label=True)
        returns_1 = labels[labels["bin"] == 1]["ret"]
        returns_0 = labels[labels["bin"] == 0]["ret"]

        assert not labels.empty
        assert len(returns_1) > 10
        assert returns_1.mean() > 0
        assert returns_1.mean() > returns_0.mean()
