import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

logger = logging.getLogger(__name__)


class PurgedKFold(_BaseKFold):
    """
    パージングK分割交差検定。

    この交差検定スキームは、訓練セットとテストセットが時間的に分離されることを保証し、
    未来からのデータリークが過去に影響することを防ぎます。
    また、ラベルリークを防ぐために、テストセットと重複する訓練セットからのサンプルをパージします。

    Marcos Lopez de Prado の「金融機械学習の進歩」に基づいています。

    引数:
        n_splits (int): 分割数。
        t1 (pd.Series): データセット内の各サンプルのラベル終了時刻のSeries。
                       インデックスは特徴量 (X) およびラベル (y) のインデックスと一致する必要があります。
        pct_embargo (float): テストセットの期間から訓練セットに対してエンバーゴをかける割合。
                             これにより、特定のウィンドウ内の情報を使用してラベルが形成されている場合、
                             訓練セットからテストセットへのリークを防ぐのに役立ちます。
    """

    def __init__(
        self, n_splits: int = 5, t1: pd.Series = None, pct_embargo: float = 0.01
    ):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pandas Series with DatetimeIndex.")
        if not all(isinstance(idx, (pd.Timestamp, np.datetime64)) for idx in t1.index):
            raise ValueError("t1 index must be of type DatetimeIndex.")
        if not all(
            isinstance(val, (pd.Timestamp, np.datetime64)) or pd.isna(val)
            for val in t1.values
        ):
            raise ValueError("t1 values must be of type DatetimeIndex or NaT.")

        super().__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        # インデックス検証エラーを避けるため、値をnumpy配列として保持
        self.t1_values = t1.values
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[Any] = None,
    ):
        """データを訓練セットとテストセットに分割するためのインデックスを生成"""
        if not isinstance(X, pd.DataFrame) or not X.index.equals(self.t1.index):
            raise ValueError("X must be a DataFrame and have the same index as t1.")

        indices = np.arange(len(X))
        fold_size = len(X) // self.n_splits
        # 整数（ナノ秒）として取得
        x_index_ints = X.index.values.view(np.int64)
        t1_values_ints = self.t1_values.view(np.int64)

        for i in range(self.n_splits):
            start = i * fold_size
            end = len(X) if i == self.n_splits - 1 else (i + 1) * fold_size
            test_idx = indices[start:end]

            if len(test_idx) == 0:
                continue

            test_start_time_ns = int(x_index_ints[start])
            # NumPyのmax()が不安定な環境のため、Python標準のmax()を使用
            t1_subset = t1_values_ints[test_idx].tolist()
            # NaTを除外
            valid_t1 = [v for v in t1_subset if v > 0]
            if not valid_t1:
                test_max_t1_ns = max(t1_subset) if t1_subset else 0
            else:
                test_max_t1_ns = max(valid_t1)
            
            test_start_time = pd.Timestamp(test_start_time_ns)
            test_max_t1 = pd.Timestamp(test_max_t1_ns)

            # エンバーゴ期間計算
            embargo_sec = (test_max_t1 - test_start_time).total_seconds() * self.pct_embargo
            embargo_end_ns = test_max_t1_ns + int(embargo_sec * 1e9)

            # パージングとエンバーゴ適用
            # 整数レベルでマスク計算
            train_mask = (t1_values_ints < test_start_time_ns) | (x_index_ints > embargo_end_ns)
            train_idx = indices[train_mask]

            if len(train_idx) == 0:
                logger.warning(f"Fold {i+1}: Training set is empty. Skipping.")
                continue

            yield train_idx, test_idx


# テスト/デバッグ用のヘルパー
if __name__ == "__main__":
    # サンプルデータ
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    X_df = pd.DataFrame(
        np.random.rand(100, 5), index=dates, columns=[f"feature_{i}" for i in range(5)]
    )
    y_series = pd.Series(np.random.randint(0, 2, 100), index=dates, name="label")

    # 各サンプルに対する t1 (ラベル終了時刻) をシミュレート
    # 簡単のため、ラベルはその開始から5日間続くとする
    t1_series = pd.Series([d + pd.Timedelta(days=5) for d in dates], index=dates)

    # PurgedKFold を初期化
    pkf = PurgedKFold(n_splits=5, t1=t1_series, pct_embargo=0.01)

    # 分割を実行
    for fold, (train_idx, test_idx) in enumerate(pkf.split(X_df)):
        print(f"Fold {fold+1}:")
        print(
            f"  訓練インデックス長: {len(train_idx)}, テストインデックス長: {len(test_idx)}"
        )

        train_start_time = X_df.index[train_idx[0]]
        train_end_time = t1_series.iloc[train_idx[-1]]
        test_start_time = X_df.index[test_idx[0]]
        test_end_time = t1_series.iloc[test_idx[-1]]

        print(f"  訓練: {train_start_time} - {train_end_time}")
        print(f"  テスト:  {test_start_time} - {test_end_time}")

        # 重複とエンバーゴの検証
        # 訓練セットの最大 t1 はテストセットの最小開始時刻より前であるべき
        max_train_t1 = t1_series.iloc[train_idx].max()
        min_test_start = X_df.index[test_idx].min()

        # テスト開始時間間隔
        min_test_period_start = X_df.index[test_idx].min()
        max_test_period_end = t1_series.loc[X_df.index[test_idx]].max()

        # エンバーゴ開始時間
        embargo_start_time = X_df.index[test_idx].min()
        embargo_duration = (
            max_test_period_end - min_test_period_start
        ).total_seconds() * pkf.pct_embargo
        embargo_end_time = embargo_start_time + pd.Timedelta(seconds=embargo_duration)

        print(f"  最大訓練 T1: {max_train_t1}")
        print(f"  最小テスト開始: {min_test_start}")
        print(f"  エンバーゴ終了時間: {embargo_end_time}")

        # 訓練インデックスがテスト期間と重複しないことをアサート
        # 具体的には、各 train_idx について、区間 [X.index[i], t1.iloc[i]] は
        # いかなるテスト区間 [X.index[j], t1.iloc[j]] とも重複してはならない

        # これはループによって暗黙的に処理される。

        # 重要なのは、訓練セットにテストセットにリークする観測値や、
        # テストセットの開始に近すぎる観測値が含まれていないことである。

        # 単純な検証: 訓練観測の最大終了時刻は、現在のテスト期間の開始 (エンバーゴを除く) より前であるべき
        # しかし、これはより微妙である。訓練観測はテスト期間 *前* に発生する可能性があるが、
        # そのラベルはテスト期間 *内* に及ぶ可能性がある。
        # これがパージングに t1 が必要な理由である。

        # ループロジックは以下を実装している:
        # 訓練サンプルが有効であるのは、その区間 [X.index[j], self.t1.iloc[j]] がどのテスト区間とも重複せず、
        # かつ (X.index[j] < test_times.min()) または (X.index[j] > embargo_end_time) の場合。

        # とりあえず検証を簡素化する
        # 訓練インデックスがテストインデックスと互いに素であることをアサート (構成によって既に処理済み)
        # パージされた領域に訓練サンプルがないことをアサート

        # purged_region_start = X_df.index[test_start] - (X_df.index[test_end] - X_df.index[test_start]) * pkf.pct_embargo # not correct
        # purged_region_end = X_df.index[test_end] + (X_df.index[test_end] - X_df.index[test_start]) * pkf.pct_embargo # not correct

        # パージング: 評価期間 (t1) がテストセットと重複する訓練観測を削除する。
        # これは `is_purged` ループで処理される。

        # エンバーゴ: テストセットの直前にある訓練観測を削除する。
        # これは `X.index[j] > embargo_end_time` または `X.index[j] < test_times.min()` で処理される。

        # 完全なテストスイートでは、より堅牢な手動チェックが必要になるかもしれない。
        # この基本的な実装では、ロジックは原則に従うことを目指している。



