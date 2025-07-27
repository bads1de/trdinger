"""
新ライブラリの動作確認テスト

オートストラテジー強化で追加したライブラリの基本動作を確認します。
"""

import pytest
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta


# 新しく追加したライブラリのインポートテスト
def test_library_imports():
    """新ライブラリのインポートテスト"""
    try:
        import sklearn
        import lightgbm as lgb
        import joblib

        print(f"✅ scikit-learn version: {sklearn.__version__}")
        print(f"✅ lightgbm version: {lgb.__version__}")
        print(f"✅ joblib version: {joblib.__version__}")

        assert True, "All libraries imported successfully"

    except ImportError as e:
        pytest.fail(f"Library import failed: {e}")


def test_sklearn_basic_functionality():
    """scikit-learnの基本機能テスト"""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score

        # サンプルデータ作成
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)  # 3クラス分類

        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

        print("✅ scikit-learn basic functionality works")

    except Exception as e:
        pytest.fail(f"scikit-learn test failed: {e}")


def test_lightgbm_basic_functionality():
    """LightGBMの基本機能テスト"""
    try:
        import lightgbm as lgb

        # サンプルデータ作成
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)  # 3クラス分類

        # LightGBMデータセット作成
        train_data = lgb.Dataset(X, label=y)

        # パラメータ設定
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
        }

        # モデル学習
        model = lgb.train(
            params,
            train_data,
            num_boost_round=10,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(0)],
        )

        # 予測
        predictions = model.predict(X)

        assert predictions.shape == (100, 3)  # 100サンプル、3クラス
        assert np.all(predictions >= 0)  # 確率は非負

        print("✅ LightGBM basic functionality works")

    except Exception as e:
        pytest.fail(f"LightGBM test failed: {e}")


def test_joblib_basic_functionality():
    """joblibの基本機能テスト"""
    try:
        import joblib
        import tempfile
        import os

        # テストデータ
        test_data = {
            "model": "test_model",
            "parameters": {"param1": 1, "param2": 2},
            "timestamp": datetime.now(),
        }

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            joblib.dump(test_data, tmp_file.name)
            tmp_file.close()  # ファイルを閉じる

            # 読み込み
            loaded_data = joblib.load(tmp_file.name)

            assert loaded_data["model"] == test_data["model"]
            assert loaded_data["parameters"] == test_data["parameters"]

            # ファイル削除
            os.unlink(tmp_file.name)

        print("✅ joblib basic functionality works")

    except Exception as e:
        pytest.fail(f"joblib test failed: {e}")


def test_feature_engineering_service_import():
    """FeatureEngineeringServiceのインポートテスト"""
    try:
        from app.services.feature_engineering import FeatureEngineeringService

        service = FeatureEngineeringService()
        assert service is not None

        print("✅ FeatureEngineeringService import successful")

    except ImportError as e:
        pytest.fail(f"FeatureEngineeringService import failed: {e}")


def test_ml_training_service_import():
    """MLTrainingServiceのインポートテスト"""
    try:
        from app.services.ml.ml_training_service import MLTrainingService

        service = MLTrainingService()
        assert service is not None

        print("✅ MLTrainingService import successful")

    except ImportError as e:
        pytest.fail(f"MLTrainingService import failed: {e}")


def test_fitness_sharing_import():
    """FitnessSharingのインポートテスト"""
    try:
        from app.services.auto_strategy.engines.fitness_sharing import (
            FitnessSharing,
        )

        fitness_sharing = FitnessSharing()
        assert fitness_sharing is not None

        print("✅ FitnessSharing import successful")

    except ImportError as e:
        pytest.fail(f"FitnessSharing import failed: {e}")


def test_sample_data_creation():
    """テスト用サンプルデータの作成"""
    try:
        # OHLCVデータの作成
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="h")

        np.random.seed(42)
        price_base = 50000

        ohlcv_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": price_base + np.random.randn(len(dates)) * 1000,
                "high": price_base + np.random.randn(len(dates)) * 1000 + 500,
                "low": price_base + np.random.randn(len(dates)) * 1000 - 500,
                "close": price_base + np.random.randn(len(dates)) * 1000,
                "volume": np.random.rand(len(dates)) * 1000000,
            }
        )

        # 価格の整合性を保つ
        for i in range(len(ohlcv_data)):
            row = ohlcv_data.iloc[i]
            high = max(row["open"], row["close"]) + abs(np.random.randn()) * 100
            low = min(row["open"], row["close"]) - abs(np.random.randn()) * 100

            ohlcv_data.at[i, "high"] = high
            ohlcv_data.at[i, "low"] = low

        assert len(ohlcv_data) > 0
        assert all(
            col in ohlcv_data.columns
            for col in ["open", "high", "low", "close", "volume"]
        )

        print(f"✅ Sample OHLCV data created: {len(ohlcv_data)} records")

        return ohlcv_data

    except Exception as e:
        pytest.fail(f"Sample data creation failed: {e}")


if __name__ == "__main__":
    """テストの直接実行"""
    print("🧪 新ライブラリの動作確認テストを開始...")

    try:
        test_library_imports()
        test_sklearn_basic_functionality()
        test_lightgbm_basic_functionality()
        test_joblib_basic_functionality()
        test_feature_engineering_service_import()
        test_fitness_sharing_import()
        sample_data = test_sample_data_creation()

        print("\n🎉 すべてのテストが正常に完了しました！")
        print("新ライブラリは正常にインストールされ、基本機能が動作しています。")

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        raise
