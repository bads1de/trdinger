import pytest
import pandas as pd
import numpy as np
from app.services.ml.cross_validation.purged_kfold import PurgedKFold

class TestPurgedKFold:
    @pytest.fixture
    def sample_data(self):
        # 100日間、毎日1レコード
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        X = pd.DataFrame(np.random.randn(100, 2), index=dates)
        y = pd.Series(np.random.randint(0, 2, 100), index=dates)
        # ラベル終了時刻(t1)は各サンプルの5日後とする
        t1 = pd.Series([d + pd.Timedelta(days=5) for d in dates], index=dates)
        return X, y, t1

    def test_initialization_errors(self):
        """初期化時のエラーチェック"""
        with pytest.raises(ValueError, match="t1 must be a pandas Series"):
            PurgedKFold(t1=[])
        
        with pytest.raises(ValueError, match="t1 index must be of type DatetimeIndex"):
            PurgedKFold(t1=pd.Series([1], index=[1]))

    def test_split_index_mismatch(self, sample_data):
        """Xとt1のインデックス不一致時のエラー"""
        X, y, t1 = sample_data
        pkf = PurgedKFold(n_splits=5, t1=t1)
        # インデックスが異なるXを渡す
        X_wrong = X.copy()
        X_wrong.index = pd.date_range(start="2024-01-01", periods=100)
        with pytest.raises(ValueError, match="X must be a DataFrame and have the same index as t1"):
            next(pkf.split(X_wrong))

    def test_purging_logic(self, sample_data):
        """パージングが正しく行われているか（未来のリーク防止）"""
        X, y, t1 = sample_data
        # 2分割でテスト。最初の分割がテストセット、次が訓練セットになるような極端な例
        pkf = PurgedKFold(n_splits=2, t1=t1, pct_embargo=0.0)
        
        splits = list(pkf.split(X, y))
        
        # 1つ目のフォールド
        # テストセット: 0-49 (2023-01-01 to 2023-02-19)
        # 訓練セット: テスト期間より後のデータ。ただしテストセットのt1(終了時刻)より後である必要がある
        train_idx, test_idx = splits[0]
        
        assert len(test_idx) == 50
        assert test_idx[0] == 0
        assert test_idx[-1] == 49
        
        # 数値比較に変える。NumPyのmax/minが壊れているため標準関数を使用。
        test_max_t1_ns = max(t1.values.view(np.int64)[test_idx].tolist())
        train_min_start_ns = min(X.index.values.view(np.int64)[train_idx].tolist())
        assert train_min_start_ns > test_max_t1_ns

    def test_embargo_logic(self, sample_data):
        """エンバーゴが正しく行われているか（テスト直後のデータ削除）"""
        X, y, t1 = sample_data
        # pct_embargoを大きく設定(50%)
        pkf = PurgedKFold(n_splits=2, t1=t1, pct_embargo=0.5)
        
        splits = list(pkf.split(X, y))
        train_idx, test_idx = splits[0]
        
        test_start_time_ns = int(X.index.values.view(np.int64)[test_idx[0]])
        test_max_t1_ns = max(t1.values.view(np.int64)[test_idx].tolist())
        
        # エンバーゴ期間の計算
        embargo_sec = (test_max_t1_ns - test_start_time_ns) / 1e9 * 0.5
        embargo_end_ns = test_max_t1_ns + int(embargo_sec * 1e9)
        
        train_indices_ns = X.index.values.view(np.int64)[train_idx].tolist()
        assert all(v > embargo_end_ns for v in train_indices_ns)

    def test_no_leakage_between_train_test(self, sample_data):
        """訓練セットとテストセットに重複がないことを確認"""
        X, y, t1 = sample_data
        pkf = PurgedKFold(n_splits=5, t1=t1)
        
        for train_idx, test_idx in pkf.split(X, y):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert train_set.isdisjoint(test_set)
            
            min_test_start_ns = min(X.index.values.view(np.int64)[test_idx].tolist())
            
            # 訓練サンプルのうちテスト開始より前のもの
            train_indices_ns = X.index.values.view(np.int64)[train_idx].tolist()
            prior_train_mask = [v < min_test_start_ns for v in train_indices_ns]
            
            if any(prior_train_mask):
                # その訓練サンプルの終了時刻(t1)がテスト開始より前であること
                all_t1_ns = t1.values.view(np.int64)[train_idx].tolist()
                prior_train_t1_ns = [all_t1_ns[i] for i, m in enumerate(prior_train_mask) if m]
                assert max(prior_train_t1_ns) < min_test_start_ns

    def test_overlap_nat_handling(self):
        """t1にNaTが含まれる場合のハンドリング"""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        X = pd.DataFrame(np.random.randn(20, 2), index=dates)
        t1 = pd.Series([d + pd.Timedelta(days=2) for d in dates], index=dates)
        t1.iloc[5] = pd.NaT # 1つだけNaT
        
        pkf = PurgedKFold(n_splits=3, t1=t1)
        # 完走することを確認
        splits = list(pkf.split(X))
        assert len(splits) == 3

    def test_insufficient_data_skipping(self):
        """データが少なすぎて訓練セットが空になる場合のスキップ"""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        X = pd.DataFrame(np.random.randn(10, 2), index=dates)
        # t1を極端に長くする（100日）
        t1 = pd.Series([d + pd.Timedelta(days=100) for d in dates], index=dates)
        
        pkf = PurgedKFold(n_splits=2, t1=t1)
        # 訓練セットが空になり、警告が出てスキップされるはず
        splits = list(pkf.split(X))
        # 全フォールドで訓練データが確保できない場合、空リストになる
        assert len(splits) < 2
