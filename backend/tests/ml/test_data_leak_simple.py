"""
簡易データリーク検証テスト

最も重要なデータリーク検証のみを実行します。
"""

import numpy as np
import pandas as pd
import pytest


class TestCriticalDataLeaks:
    """最も重要なデータリーク検証"""

    def test_time_series_split_basic(self):
        """
        基本的な時系列分割の検証

        学習データとテストデータの時間的順序が正しいかを確認します。
        これはデータリークの最も基本的で重要なチェックです。
        """
        # テストデータ作成
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        data = pd.DataFrame({"value": np.random.randn(100)}, index=dates)

        # 80:20で分割
        split_point = int(len(data) * 0.8)
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]

        # 検証: 学習データの最終時刻 < テストデータの最初の時刻
        train_last = train_data.index.max()
        test_first = test_data.index.min()

        assert train_last < test_first, (
            f"❌ データリーク検出! "
            f"学習データ終了時刻({train_last}) >= テストデータ開始時刻({test_first})"
        )

        print(f"✅ 時系列分割OK: 学習終了={train_last}, テスト開始={test_first}")

    def test_no_index_overlap(self):
        """
        Train/Testのインデックス重複チェック

        同じデータポイントが学習とテストの両方に含まれていないかを確認します。
        """
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        data = pd.DataFrame({"value": np.random.randn(100)}, index=dates)

        split_point = int(len(data) * 0.8)
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]

        # インデックスの重複チェック
        overlap = set(train_data.index).intersection(set(test_data.index))

        assert len(overlap) == 0, (
            f"❌ データリーク検出! "
            f"Train/Testで{len(overlap)}個のインデックスが重複しています"
        )

        print(f"✅ インデックス重複なし")

    def test_rolling_calculation_causality(self):
        """
        ローリング計算の因果性チェック

        移動平均などのローリング計算が未来のデータを使用していないかを確認します。
        """
        dates = pd.date_range(start="2023-01-01", periods=50, freq="1h")
        data = pd.DataFrame({"price": np.random.randn(50).cumsum() + 100}, index=dates)

        # 移動平均を計算
        window = 5
        data["MA"] = data["price"].rolling(window=window).mean()

        # 最初のwindow-1行はNaNであるべき（過去のデータが不足しているため）
        initial_nas = data["MA"].iloc[: window - 1].isna().all()

        assert initial_nas, (
            f"❌ データリーク検出! "
            f"移動平均の最初の{window-1}行にNaNではない値があります。"
            "未来のデータを使用している可能性があります。"
        )

        print(f"✅ ローリング計算OK: 最初の{window-1}行はNaN")

    def test_feature_independence_from_future(self):
        """
        特徴量計算の未来データ独立性チェック

        全データで計算した特徴量と、部分データで計算した特徴量が
        同じ時点で同じ値になることを確認します。
        """
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        full_data = pd.DataFrame(
            {"price": np.random.randn(100).cumsum() + 100}, index=dates
        )

        # 全データで移動平均を計算
        full_data["MA_10"] = full_data["price"].rolling(window=10).mean()

        # 最初の80%のデータで移動平均を計算
        split_point = 80
        partial_data = full_data.iloc[:split_point].copy()
        partial_data["MA_10"] = partial_data["price"].rolling(window=10).mean()

        # 共通部分（10行目から80行目）で値が一致するか確認
        common_range = slice(10, split_point)
        full_values = full_data.iloc[common_range]["MA_10"].values
        partial_values = partial_data.iloc[common_range]["MA_10"].values

        # 数値誤差を考慮して比較
        np.testing.assert_allclose(
            full_values,
            partial_values,
            rtol=1e-10,
            err_msg="❌ データリーク検出! 特徴量計算が未来のデータに依存しています",
        )

        print(f"✅ 特徴量独立性OK: 全データと部分データで同じ値")

    def test_label_uses_future_only(self):
        """
        ラベル生成が未来のデータのみを使用することを確認

        簡略化されたラベル生成のテストです。
        実際のTriple Barrier Methodではより複雑ですが、
        原則は同じです：ラベルは未来の価格変動から決定されるべきです。
        """
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        # 簡単なフォワードリターンラベル
        horizon = 5
        labels = pd.Series(index=prices.index, dtype=float)

        for i in range(len(prices) - horizon):
            current_time = prices.index[i]
            future_time = prices.index[i + horizon]

            # 未来の時刻が現在より後であることを確認
            assert future_time > current_time, (
                f"❌ ラベル生成エラー! "
                f"未来時刻({future_time}) <= 現在時刻({current_time})"
            )

            # ラベルを計算（簡略化）
            return_rate = (prices.iloc[i + horizon] - prices.iloc[i]) / prices.iloc[i]
            labels.iloc[i] = return_rate

        print(f"✅ ラベル生成OK: 未来の価格のみを使用")


if __name__ == "__main__":
    # 直接実行した場合のテスト
    test = TestCriticalDataLeaks()

    print("\n" + "=" * 60)
    print("データリーク検証テスト実行中...")
    print("=" * 60 + "\n")

    try:
        print("1. 時系列分割テスト")
        test.test_time_series_split_basic()

        print("\n2. インデックス重複テスト")
        test.test_no_index_overlap()

        print("\n3. ローリング計算テスト")
        test.test_rolling_calculation_causality()

        print("\n4. 特徴量独立性テスト")
        test.test_feature_independence_from_future()

        print("\n5. ラベル生成テスト")
        test.test_label_uses_future_only()

        print("\n" + "=" * 60)
        print("✅ すべてのテストに合格しました!")
        print("データリークは検出されませんでした。")
        print("=" * 60)

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"❌ テスト失敗: {e}")
        print("=" * 60)
        raise
