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

            test_start_time_ns = x_index_ints[start]

            # --- 高速化箇所: NumPyで最大値を取得 ---
            t1_subset_ints = t1_values_ints[test_idx]
            # NaT は負の値（通常 -9223372036854775808）なので、maxで正しく最新時刻が取れる
            # すべてNaTの場合は 0 または負の値になる
            test_max_t1_ns = np.max(t1_subset_ints)

            if test_max_t1_ns < 0:  # 全てNaTの場合
                test_max_t1_ns = x_index_ints[end - 1]  # テストセットの末尾を使用

            # エンバーゴ期間計算 (ベクトル化)
            embargo_sec = (
                (test_max_t1_ns - test_start_time_ns) / 1e9
            ) * self.pct_embargo
            embargo_end_ns = test_max_t1_ns + int(embargo_sec * 1e9)

            # パージングとエンバーゴ適用
            # 整数レベルでマスク計算 (ベクトル化済み)
            train_mask = (t1_values_ints < test_start_time_ns) | (
                x_index_ints > embargo_end_ns
            )
            train_idx = indices[train_mask]

            if len(train_idx) == 0:
                logger.warning(f"Fold {i+1}: Training set is empty. Skipping.")
                continue

            yield train_idx, test_idx


# テスト/デバッグ用のヘルパー
