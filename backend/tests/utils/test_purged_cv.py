import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.utils.purged_cv import PurgedKFold

class TestPurgedKFold:
    def setup_method(self):
        # テスト用データフレームを作成
        dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="D"))
        self.X = pd.DataFrame(np.random.rand(100, 5), index=dates, columns=[f'feature_{i}' for i in range(5)])
        self.y = pd.Series(np.random.randint(0, 2, 100), index=dates)
        # t1 (ラベルの終了時刻) を模擬したSeries。X.indexと同じインデックスを持つ。
        # ここでは単純にX.indexに数日加算したものをt1とする。
        self.t1_series = pd.Series(dates + pd.Timedelta(days=5), index=dates)

    def test_initialization(self):
        """初期化テスト"""
        cv = PurgedKFold(n_splits=5, t1=self.t1_series, embargo_pct=0.01)
        assert cv.n_splits == 5
        assert isinstance(cv.t1, pd.Series)
        assert cv.embargo_pct == 0.01

    def test_split_counts(self):
        """分割数が正しいか"""
        cv = PurgedKFold(n_splits=5, t1=self.t1_series, embargo_pct=0.01)
        folds = list(cv.split(self.X, self.y))
        assert len(folds) == 5

    def test_no_leakage(self):
        """リークがないか（PurgingとEmbargoの確認）"""
        # embargo_pctを大きくしてリークを強調
        cv = PurgedKFold(n_splits=5, t1=self.t1_series, embargo_pct=0.1) # 10% embargo

    def test_no_leakage(self):
        """リークがないか（PurgingとEmbargoの確認）"""
        # embargo_pctを大きくしてリークを強調
        cv = PurgedKFold(n_splits=5, t1=self.t1_series, embargo_pct=0.5) # 50% embargo for clear test

        folds = list(cv.split(self.X, self.y))
        
        for i, (train_idx, test_idx) in enumerate(folds):
            if len(test_idx) == 0:
                continue

            test_start_t0 = self.X.index[test_idx[0]]
            test_end_t1 = self.t1_series.iloc[test_idx[-1]]

            # Embargo period calculation matching PurgedKFold's split method
            test_observation_span = self.X.index[test_idx[-1]] - self.X.index[test_idx[0]]
            embargo_period_duration = test_observation_span * cv.embargo_pct
            min_train_t0_after_embargo = test_end_t1 + embargo_period_duration

            # 1. Test set should not overlap with train set
            assert len(np.intersect1d(train_idx, test_idx)) == 0, f"Fold {i}: Train and Test indices overlap directly."

            # 2. Purging check: No training label should overlap with the test period
            # For each train_j in train_idx, its label period [X.index[j], t1.iloc[j]]
            # should not intersect with [test_start_t0, test_end_t1]
            for j in train_idx:
                train_t0_j = self.X.index[j]
                train_t1_j = self.t1_series.iloc[j]
                
                # Check for intersection: (start1 <= end2) and (end1 >= start2)
                has_intersection = (train_t0_j <= test_end_t1) and (train_t1_j >= test_start_t0)
                assert not has_intersection, \
                    f"Fold {i}: Purging failed: Train label [{train_t0_j}, {train_t1_j}] overlaps with Test period [{test_start_t0}, {test_end_t1}]"

            # 3. Embargo check: No training observation should start within the embargo period
            # For each train_j in train_idx, its observation start time X.index[j]
            # should not be within [test_end_t1, min_train_t0_after_embargo]
            for j in train_idx:
                train_t0_j = self.X.index[j]
                is_within_embargo = (train_t0_j >= test_end_t1) and (train_t0_j <= min_train_t0_after_embargo)
                assert not is_within_embargo, \
                    f"Fold {i}: Embargo failed: Train observation starts at {train_t0_j} within embargo period [{test_end_t1}, {min_train_t0_after_embargo}]"

            # 4. Ensure that the training indices are strictly before the test indices (overall time order)
            # PurgedKFoldでは訓練データがテストデータより「厳密に前」である必要はないため、このアサーションは削除する。
            # 重要なのは、訓練データのラベル期間が、テストデータの観測期間やエンバーゴ期間と重ならないこと。
            # if len(train_idx) > 0 and len(test_idx) > 0:
            #     assert train_idx.max() < test_idx.min(), \
            #         f"Fold {i}: Training indices not strictly before test indices. Max train: {train_idx.max()}, Min test: {test_idx.min()}"


