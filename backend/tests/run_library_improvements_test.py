"""
3.9と3.10の修正内容をテストするためのスクリプト
"""

import sys
import os
import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
backend_app_path = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, backend_app_path)


def test_optimized_crypto_features():
    """OptimizedCryptoFeaturesのテスト"""
    print("=== OptimizedCryptoFeaturesのテスト ===")

    try:
        from services.ml.feature_engineering.optimized_crypto_features import (
            OptimizedCryptoFeatures,
        )

        # テストデータ作成
        dates = pd.date_range("2023-01-01", periods=50, freq="H")
        np.random.seed(42)

        test_data = pd.DataFrame(
            {
                "Open": 100 + np.random.randn(50) * 5,
                "High": 105 + np.random.randn(50) * 5,
                "Low": 95 + np.random.randn(50) * 5,
                "Close": 100 + np.random.randn(50) * 5,
                "Volume": 1000 + np.random.randn(50) * 100,
                "open_interest": 5000 + np.random.randn(50) * 500,
                "funding_rate": np.random.randn(50) * 0.001,
                "fear_greed_value": 50 + np.random.randn(50) * 20,
            },
            index=dates,
        )

        # 特徴量エンジンのテスト
        feature_engine = OptimizedCryptoFeatures()
        result = feature_engine.create_optimized_features(test_data)

        # 基本的な検証
        assert isinstance(result, pd.DataFrame), "結果がDataFrameではありません"
        assert len(result) == len(test_data), "行数が一致しません"
        assert len(result.columns) > len(
            test_data.columns
        ), "特徴量が追加されていません"

        # 無限値やNaN値のチェック
        infinite_check = result.isin([np.inf, -np.inf]).any().any()
        assert not infinite_check, "無限値が含まれています"

        # ロバストリターン特徴量のチェック
        robust_return_cols = [col for col in result.columns if "robust_return" in col]
        assert len(robust_return_cols) > 0, "ロバストリターン特徴量が作成されていません"

        for col in robust_return_cols:
            assert np.isfinite(result[col]).all(), f"{col}に無限値が含まれています"

        print("✅ OptimizedCryptoFeaturesのテスト成功")
        return True

    except Exception as e:
        print(f"❌ OptimizedCryptoFeaturesのテスト失敗: {e}")
        return False


def test_data_validator():
    """DataValidatorのテスト"""
    print("=== DataValidatorのテスト ===")

    try:
        from utils.data_validation import DataValidator

        # テストデータ作成
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # 正規化実行
        normalized = DataValidator.safe_normalize(data, window=5)

        # 基本的な検証
        assert isinstance(normalized, pd.Series), "結果がSeriesではありません"
        assert np.isfinite(normalized).all(), "無限値やNaN値が含まれています"

        # 定数値での正規化テスト
        constant_data = pd.Series([5, 5, 5, 5, 5])
        normalized_constant = DataValidator.safe_normalize(constant_data, window=3)
        assert np.isfinite(
            normalized_constant
        ).all(), "定数値正規化で無限値が発生しました"

        print("✅ DataValidatorのテスト成功")
        return True

    except Exception as e:
        print(f"❌ DataValidatorのテスト失敗: {e}")
        return False


def test_knn_model():
    """KNNModelのテスト"""
    print("=== KNNModelのテスト ===")

    try:
        from services.ml.models.knn_wrapper import KNNModel

        # テストデータ作成
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(50, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        y_train = pd.Series(np.random.randint(0, 2, 50))
        X_test = pd.DataFrame(
            np.random.randn(10, 5), columns=[f"feature_{i}" for i in range(5)]
        )

        # モデル初期化
        model = KNNModel(n_neighbors=5, metric="euclidean")

        # デフォルトパラメータのチェック
        assert "leaf_size" in model.default_params, "leaf_sizeが含まれていません"
        assert (
            model.default_params["leaf_size"] == 30
        ), "leaf_sizeの値が正しくありません"

        # 学習実行
        model.fit(X_train, y_train)

        # 学習状態のチェック
        assert model.is_trained, "学習が完了していません"
        assert model.model is not None, "モデルが初期化されていません"

        # 予測実行
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test), "予測結果の長さが正しくありません"

        print("✅ KNNModelのテスト成功")
        return True

    except Exception as e:
        print(f"❌ KNNModelのテスト失敗: {e}")
        return False


def test_ensemble_parameter_space():
    """EnsembleParameterSpaceのテスト"""
    print("=== EnsembleParameterSpaceのテスト ===")

    try:
        from services.optimization.ensemble_parameter_space import (
            EnsembleParameterSpace,
        )

        # KNNパラメータ空間のテスト
        param_space = EnsembleParameterSpace.get_knn_parameter_space()

        # 新しいmetricパラメータのチェック
        assert "knn_metric" in param_space, "knn_metricパラメータが追加されていません"

        # metricの選択肢のチェック
        metric_categories = param_space["knn_metric"].categories
        expected_metrics = ["minkowski", "euclidean", "manhattan", "chebyshev"]
        for metric in expected_metrics:
            assert metric in metric_categories, f"{metric}が選択肢に含まれていません"

        print("✅ EnsembleParameterSpaceのテスト成功")
        return True

    except Exception as e:
        print(f"❌ EnsembleParameterSpaceのテスト失敗: {e}")
        return False


def main():
    """メインテスト実行"""
    print("3.9と3.10の修正内容テストを開始します...\n")

    results = []

    # 各テストを実行
    results.append(test_optimized_crypto_features())
    results.append(test_data_validator())
    results.append(test_knn_model())
    results.append(test_ensemble_parameter_space())

    # 結果サマリー
    print("\n=== テスト結果サマリー ===")
    success_count = sum(results)
    total_count = len(results)

    print(f"成功: {success_count}/{total_count}")

    if success_count == total_count:
        print("🎉 すべてのテストが成功しました！")
        return True
    else:
        print("⚠️ 一部のテストが失敗しました。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
