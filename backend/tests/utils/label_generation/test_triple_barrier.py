import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.services.ml.label_generation.triple_barrier import TripleBarrier

class TestTripleBarrier:
    """トリプルバリアラベル生成のテストクラス"""

    def setup_method(self):
        # 1時間ごとのデータ
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        self.close = pd.Series(100.0, index=dates) # 初期値100
        # 変動を加える (ランダムウォーク)
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100) # 1%の変動
        self.close = self.close * (1 + returns).cumprod()
        
        self.volatility = pd.Series(0.01, index=dates) # ボラティリティ 1% (一定と仮定)

    def test_profit_taking(self):
        """利確バリアにヒットするケース"""
        # テスト用にデータを加工: index[0]から見て、index[5]で確実に上がるようにする
        # index[0] = 100.0 と仮定
        self.close.iloc[0] = 100.0
        self.close.iloc[1:5] = 100.0 # 途中は動かない
        self.close.iloc[5] = 103.0 # 5時間後に +3%
        
        tb = TripleBarrier(
            num_threads=1
        )
        
        # 垂直バリアは10時間後
        vertical_barriers = pd.Series(self.close.index + pd.Timedelta(hours=10), index=self.close.index)
        
        # pt=2.0 => 2.0 * vol(1%) = 2% 上昇で利確
        events = tb.get_events(
            close=self.close,
            t_events=self.close.index[:10], # 最初の10個だけテスト
            pt_sl=[2.0, 2.0],
            target=self.volatility,
            min_ret=0.001,
            vertical_barrier_times=vertical_barriers
        )
        
        # ラベル生成
        labels = tb.get_bins(events, self.close)
        
        # index[0] は +3% (index[5]) で利確ライン(+2%)を超えるはず
        assert labels.loc[self.close.index[0], 'bin'] == 1.0
        
    def test_stop_loss(self):
        """損切りバリアにヒットするケース"""
        self.close.iloc[0] = 100.0
        self.close.iloc[1:5] = 100.0
        self.close.iloc[5] = 97.0 # 5時間後に -3%
        
        tb = TripleBarrier(num_threads=1)
        
        vertical_barriers = pd.Series(self.close.index + pd.Timedelta(hours=10), index=self.close.index)
        
        events = tb.get_events(
            close=self.close,
            t_events=self.close.index[:10],
            pt_sl=[2.0, 2.0], # 2% 下落で損切り
            target=self.volatility,
            min_ret=0.001,
            vertical_barrier_times=vertical_barriers
        )
        
        labels = tb.get_bins(events, self.close)
        
        # index[0] は -3% で損切りライン(-2%)に触れる -> -1
        assert labels.loc[self.close.index[0], 'bin'] == -1.0

    def test_time_barrier(self):
        """時間切れのケース"""
        self.close.iloc[0] = 100.0
        self.close.iloc[1:10] = 100.5 # +0.5% (バリア 2% に届かない)
        
        tb = TripleBarrier(num_threads=1)
        
        # 垂直バリアを2時間後に設定
        vertical_barriers = pd.Series(self.close.index + pd.Timedelta(hours=2), index=self.close.index)
        
        events = tb.get_events(
            close=self.close,
            t_events=self.close.index[:10],
            pt_sl=[2.0, 2.0],
            target=self.volatility,
            min_ret=0.001,
            vertical_barrier_times=vertical_barriers
        )
        
        labels = tb.get_bins(events, self.close)
        
        # バリアに触れずに時間切れ -> 0
        assert labels.loc[self.close.index[0], 'bin'] == 0.0

