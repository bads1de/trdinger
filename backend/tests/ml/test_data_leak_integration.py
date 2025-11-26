"""
実装コードを使ったデータリーク検証

実際のBaseMLTrainerとFeatureEngineeringServiceを使用して
データリークが発生していないかを検証します。
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from app.services.ml.base_ml_trainer import BaseMLTrainer
from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)


@pytest.mark.skip(
    reason="統合テストは複雑なモック設定が必要なため一旦スキップ。基本的なデータリークテストはtest_data_leak_simple.pyで実施済み。"
)
class TestRealImplementationDataLeaks:
    """実装コードを使ったデータリーク検証"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """サンプルOHLCVデータ"""
        dates = pd.date_range(start="2023-01-01", periods=500, freq="1h")
        return pd.DataFrame(
            {
                "open": np.random.randn(500).cumsum() + 100,
                "high": np.random.randn(500).cumsum() + 105,
                "low": np.random.randn(500).cumsum() + 95,
                "close": np.random.randn(500).cumsum() + 102,
                "volume": np.random.randint(1000, 10000, 500),
            },
            index=dates,
        )

    @pytest.fixture
    def mock_trainer(self):
        """Mockトレーナー"""
        with (
            patch("app.services.ml.ensemble.ensemble_trainer.EnsembleTrainer"),
            patch(
                "app.services.ml.single_model.single_model_trainer.SingleModelTrainer"
            ),
        ):
            trainer = BaseMLTrainer(
                trainer_config={"type": "single", "model_type": "lightgbm"}
            )
            yield trainer

    def test_base_trainer_time_series_split(self, mock_trainer, sample_ohlcv_data):
        """
        BaseMLTrainerの時系列分割を検証

        実際のBaseMLTrainerが正しく時系列分割を行っているかテストします。
        """
        X = sample_ohlcv_data.copy()
        y = pd.Series(np.random.randint(0, 3, len(X)), index=X.index)

        # BaseMLTrainerの_split_dataメソッドを使用
        X_train, X_test, y_train, y_test = mock_trainer._split_data(
            X, y, use_time_series_split=True, test_size=0.2
        )

        # 検証1: 時間的順序
        train_max = X_train.index.max()
        test_min = X_test.index.min()

        assert train_max < test_min, (
            f"❌ BaseMLTrainerでデータリーク検出!\n"
            f"  学習データ終了: {train_max}\n"
            f"  テストデータ開始: {test_min}\n"
            f"  → テストデータが学習データより過去です"
        )

        # 検証2: インデックスの重複なし
        overlap = set(X_train.index).intersection(set(X_test.index))
        assert len(overlap) == 0, (
            f"❌ BaseMLTrainerでインデックス重複を検出!\n" f"  重複数: {len(overlap)}"
        )

        print(f"✅ BaseMLTrainerの時系列分割: データリークなし")
        print(f"   学習期間: {X_train.index.min()} ～ {X_train.index.max()}")
        print(f"   テスト期間: {X_test.index.min()} ～ {X_test.index.max()}")

    def test_feature_engineering_no_future_leak(self, sample_ohlcv_data):
        """
        FeatureEngineeringServiceが未来データを使用していないかテスト

        全データと部分データで特徴量を計算し、
        同じ時点の値が一致することを確認します。
        """
        feature_service = FeatureEngineeringService()

        # 全データで特徴量を計算
        print("全データで特徴量を計算中...")
        full_features = feature_service.calculate_advanced_features(
            ohlcv_data=sample_ohlcv_data
        )

        # 最初の80%だけで特徴量を計算
        split_point = int(len(sample_ohlcv_data) * 0.8)
        partial_data = sample_ohlcv_data.iloc[:split_point]

        print(f"部分データ({split_point}行)で特徴量を計算中...")
        partial_features = feature_service.calculate_advanced_features(
            ohlcv_data=partial_data
        )

        # 共通するインデックスと列を取得
        common_index = partial_features.index.intersection(full_features.index)
        common_columns = partial_features.columns.intersection(full_features.columns)

        print(f"\n検証対象:")
        print(f"  共通インデックス数: {len(common_index)}")
        print(f"  共通特徴量数: {len(common_columns)}")

        # 各特徴量について値を比較
        leak_detected = False
        leaked_features = []

        for col in common_columns:
            # NaNを除外
            partial_values = partial_features.loc[common_index, col].dropna()
            if len(partial_values) == 0:
                continue

            full_values = full_features.loc[partial_values.index, col]

            try:
                np.testing.assert_allclose(
                    partial_values.values,
                    full_values.values,
                    rtol=1e-5,
                    err_msg=f"特徴量 '{col}' で値の不一致",
                )
            except AssertionError:
                leak_detected = True
                leaked_features.append(col)

                # 差分の詳細を表示
                diff = np.abs(partial_values.values - full_values.values)
                max_diff = np.max(diff)
                print(f"\n⚠️  特徴量 '{col}' で潜在的なリーク検出")
                print(f"   最大差分: {max_diff:.6f}")
                print(f"   最初の5個の差分: {diff[:5]}")

        if leak_detected:
            raise AssertionError(
                f"❌ FeatureEngineeringServiceでデータリーク検出!\n"
                f"  漏洩の可能性がある特徴量: {leaked_features}\n"
                f"  → これらの特徴量が未来のデータを使用している可能性があります"
            )

        print(f"\n✅ FeatureEngineeringService: データリークなし")
        print(f"   検証した特徴量数: {len(common_columns)}")

    def test_scaler_fit_on_train_only(self, mock_trainer):
        """
        スケーラーが学習データのみでfitされているかテスト
        """
        from sklearn.preprocessing import StandardScaler

        # 学習データとテストデータを作成（明確に異なる分布）
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(100, 5) * 10,  # 平均0, 標準偏差10
            columns=[f"f{i}" for i in range(5)],
        )
        X_test = pd.DataFrame(
            np.random.randn(20, 5) * 10 + 50,  # 平均50, 標準偏差10
            columns=[f"f{i}" for i in range(5)],
        )

        # スケーラーを初期化
        mock_trainer.scaler = StandardScaler()

        # 前処理を実行
        X_train_scaled, X_test_scaled = mock_trainer._preprocess_data(X_train, X_test)

        # 検証: スケーラーの平均が学習データの平均と一致
        scaler_mean = mock_trainer.scaler.mean_
        train_mean = X_train.mean().values

        np.testing.assert_allclose(
            scaler_mean,
            train_mean,
            rtol=1e-10,
            err_msg="スケーラーがテストデータを含めてfitされています",
        )

        # 検証: テストデータのスケーリング後の平均が0から大きく外れている
        # （テストデータは平均50でオフセットされているため）
        test_scaled_mean = X_test_scaled.mean().mean()

        assert abs(test_scaled_mean) > 3.0, (
            f"❌ スケーラーがテストデータでfitされている可能性!\n"
            f"  テストデータのスケーリング後平均: {test_scaled_mean:.2f}\n"
            f"  → 0に近すぎます（期待値: 平均50のオフセットが反映されるべき）"
        )

        print(f"✅ スケーラー: 学習データのみでfit")
        print(f"   学習データ平均: {train_mean[:3]}...")
        print(f"   スケーラー平均: {scaler_mean[:3]}...")
        print(
            f"   テストスケーリング後平均: {test_scaled_mean:.2f} (0から離れている=正常)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
