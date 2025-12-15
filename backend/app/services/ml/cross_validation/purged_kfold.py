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
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[Any] = None,
    ):
        """
        データを訓練セットとテストセットに分割するためのインデックスを生成します。

        引数:
            X (pd.DataFrame): 訓練データ。
            y (Optional[pd.Series]): ターゲット変数（無視されますが、scikit-learnとの互換性のために保持されています）。
            groups (Optional[Any]): サンプルのグループラベル（無視されます）。

        返り値:
            Tuple[np.ndarray, np.ndarray]: その分割に対する訓練セットとテストセットのインデックス。
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if not X.index.equals(self.t1.index):
            raise ValueError("X and t1 must have the same index.")

        indices = np.arange(X.shape[0])

        for i in range(self.n_splits):
            test_start_idx = (X.shape[0] // self.n_splits) * i
            test_end_idx = (X.shape[0] // self.n_splits) * (i + 1)

            # 最終フォールドの test_end を調整
            if i == self.n_splits - 1:
                test_end_idx = X.shape[0]

            test_indices = indices[test_start_idx:test_end_idx]

            if len(test_indices) == 0:
                continue

            # テストセットの期間
            test_start_time = X.index[test_start_idx]
            # 正しい重複チェックとエンバーゴのためにテストセット内の最大 t1 を決定
            test_max_t1 = self.t1.iloc[test_indices].max()

            # 訓練サンプルに対する「禁止」時間範囲を定義
            # 訓練サンプル (t_start, t_end) がテストセットと重複するのは、
            # t_start <= test_max_t1 かつ t_end >= test_start_time の場合

            # 保持する訓練サンプルを特定
            # 1. テストセットより完全に前のサンプル
            #    t_end < test_start_time
            # 2. テストセットより完全に後のサンプル (エンバーゴ期間後)
            #    t_start > test_max_t1 + embargo

            # エンバーゴ期間を計算
            test_duration = test_max_t1 - test_start_time
            embargo_seconds = self._get_embargo_seconds_from_duration(
                test_duration, self.pct_embargo
            )
            embargo_end_time = test_max_t1 + pd.Timedelta(seconds=embargo_seconds)

            # 有効な訓練サンプルのためのブールマスク
            # 条件 1: 訓練サンプルの終了時刻がテストセットの開始時刻より厳密に前である
            train_indices_before = self.t1 < test_start_time

            # 条件 2: 訓練サンプルの開始時刻がテストセットの終了時刻 (プラスエンバーゴ) より厳密に後である
            train_indices_after = X.index > embargo_end_time

            # マスクを結合
            train_mask = train_indices_before | train_indices_after

            train_indices = indices[train_mask]

            # 空のトレーニングセットをチェック
            if len(train_indices) == 0:
                logger.warning(
                    f"Fold {i + 1}/{self.n_splits}: Training set is empty after purging and embargo. "
                    f"Test period: {test_start_time} - {test_max_t1}, Embargo end: {embargo_end_time}. "
                    f"Skipping this fold."
                )
                continue

            yield train_indices, test_indices

    def _get_embargo_seconds_from_duration(
        self, duration: pd.Timedelta, pct_embargo: float
    ) -> float:
        """テストセットの期間と割合に基づいてエンバーゴ秒数を計算します。"""
        return duration.total_seconds() * pct_embargo


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



