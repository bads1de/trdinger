import time
import numpy as np
import pandas as pd
import pytest
from app.services.ml.cross_validation.purged_kfold import PurgedKFold

class TestPurgedKFoldPerformance:
    def test_performance_large_dataset(self):
        """大規模データセットでのパフォーマンスをテスト"""
        n_samples = 5000  # CI/CDでの過度な待機時間を避けるため10000から削減したが、N^2の問題を示すには十分
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="h")
        X = pd.DataFrame(np.random.randn(n_samples, 5), index=dates, columns=[f"feat_{i}" for i in range(5)])
        
        # 未来24時間にまたがるラベルをシミュレート
        t1 = pd.Series(dates + pd.Timedelta(hours=24), index=dates)
        
        pkf = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
        
        start_time = time.time()
        # ジェネレータを消費して強制的に実行させる
        list(pkf.split(X))
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\nパフォーマンステスト (N={n_samples}): {duration:.4f} 秒")
        
        # 現在のO(N^2)実装では、マシンによっては5000サンプルで1秒以上かかる可能性がある
        # より高速な実行を目指す