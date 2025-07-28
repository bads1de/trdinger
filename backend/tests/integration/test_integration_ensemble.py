"""
アンサンブル学習統合テスト

MLTrainingServiceとEnsembleTrainerの統合動作を確認します。
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.ml.ml_training_service import MLTrainingService


def create_test_data():
    """テスト用データを作成"""
    print("統合テスト用データを作成中...")

    # 分類データセットを生成
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # DataFrameに変換（OHLCVカラムを含む）
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    data = pd.DataFrame(X, columns=feature_names)

    # 必要なOHLCVカラムを追加（ダミーデータ）
    data["Open"] = np.random.uniform(100, 200, len(data))
    data["High"] = data["Open"] + np.random.uniform(0, 10, len(data))
    data["Low"] = data["Open"] - np.random.uniform(0, 10, len(data))
    data["Close"] = data["Open"] + np.random.uniform(-5, 5, len(data))
    data["Volume"] = np.random.uniform(1000, 10000, len(data))
    data["target"] = y

    print(f"統合テストデータ作成完了: {len(data)}行, {len(feature_names)}特徴量")
    print(f"クラス分布: {pd.Series(y).value_counts().to_dict()}")

    return data


def test_ml_training_service_ensemble():
    """MLTrainingServiceでアンサンブル学習テスト"""
    print("\n=== MLTrainingService アンサンブル学習統合テスト ===")

    try:
        # テストデータを作成
        training_data = create_test_data()

        # アンサンブル設定付きMLTrainingServiceを作成
        ensemble_config = {
            "method": "bagging",
            "bagging_params": {
                "n_estimators": 3,
                "bootstrap_fraction": 0.8,
                "base_model_type": "lightgbm",
            },
        }

        ml_service = MLTrainingService(
            trainer_type="ensemble", ensemble_config=ensemble_config
        )

        print(f"MLTrainingService作成完了: trainer_type={ml_service.trainer_type}")
        print(f"アンサンブル設定: {ml_service.ensemble_config}")

        # 学習実行
        print("\nアンサンブル学習を実行中...")
        result = ml_service.train_model(
            training_data=training_data,
            save_model=False,
            test_size=0.2,
            random_state=42,
        )

        print(f"学習完了!")
        print(f"成功: {result.get('success', False)}")
        accuracy = result.get("accuracy", "N/A")
        if isinstance(accuracy, (int, float)):
            print(f"精度: {accuracy:.4f}")
        else:
            print(f"精度: {accuracy}")
        print(f"モデルタイプ: {result.get('model_type', 'N/A')}")
        print(f"アンサンブル手法: {result.get('ensemble_method', 'N/A')}")
        print(f"学習サンプル数: {result.get('train_samples', 'N/A')}")
        print(f"テストサンプル数: {result.get('test_samples', 'N/A')}")

        # 予測テスト
        if result.get("success", False):
            print("\n予測テストを実行中...")
            test_features = training_data.drop("target", axis=1).head(10)
            predictions = ml_service.predict(test_features)

            print(f"予測結果形状: {predictions.shape}")
            print(f"予測サンプル: {predictions[:3]}")

        return result.get("success", False)

    except Exception as e:
        print(f"MLTrainingService統合テストでエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_stacking_integration():
    """スタッキング統合テスト"""
    print("\n=== スタッキング統合テスト ===")

    try:
        # テストデータを作成
        training_data = create_test_data()

        # スタッキング設定付きMLTrainingServiceを作成
        ensemble_config = {
            "method": "stacking",
            "stacking_params": {
                "base_models": ["lightgbm", "random_forest"],
                "meta_model": "lightgbm",
                "cv_folds": 3,
                "use_probas": True,
            },
        }

        ml_service = MLTrainingService(
            trainer_type="ensemble", ensemble_config=ensemble_config
        )

        print(f"スタッキング設定: {ml_service.ensemble_config}")

        # 学習実行
        print("\nスタッキング学習を実行中...")
        result = ml_service.train_model(
            training_data=training_data,
            save_model=False,
            test_size=0.2,
            random_state=42,
        )

        print(f"スタッキング学習完了!")
        print(f"成功: {result.get('success', False)}")
        accuracy = result.get("accuracy", "N/A")
        if isinstance(accuracy, (int, float)):
            print(f"精度: {accuracy:.4f}")
        else:
            print(f"精度: {accuracy}")
        print(f"アンサンブル手法: {result.get('ensemble_method', 'N/A')}")
        print(f"ベースモデル数: {len(result.get('base_model_results', []))}")

        return result.get("success", False)

    except Exception as e:
        print(f"スタッキング統合テストでエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_automl_ensemble_integration():
    """AutoML + アンサンブル統合テスト"""
    print("\n=== AutoML + アンサンブル統合テスト ===")

    try:
        # テストデータを作成
        training_data = create_test_data()

        # AutoML設定
        automl_config = {
            "enabled": True,
            "feature_selection": {
                "enabled": True,
                "method": "mutual_info",
                "k_best": 10,
            },
            "feature_engineering": {
                "enabled": True,
                "polynomial_features": False,
                "interaction_features": False,
            },
        }

        # AutoML + アンサンブル設定
        ensemble_config = {
            "method": "bagging",
            "bagging_params": {
                "n_estimators": 3,
                "bootstrap_fraction": 0.8,
                "base_model_type": "lightgbm",
            },
        }

        ml_service = MLTrainingService(
            trainer_type="ensemble",
            automl_config=automl_config,
            ensemble_config=ensemble_config,
        )

        print(f"AutoML + アンサンブル設定完了")
        print(f"AutoML有効: {automl_config['enabled']}")
        print(f"アンサンブル手法: {ensemble_config['method']}")

        # 学習実行
        print("\nAutoML + アンサンブル学習を実行中...")
        result = ml_service.train_model(
            training_data=training_data,
            save_model=False,
            test_size=0.2,
            random_state=42,
        )

        print(f"AutoML + アンサンブル学習完了!")
        print(f"成功: {result.get('success', False)}")
        accuracy = result.get("accuracy", "N/A")
        if isinstance(accuracy, (int, float)):
            print(f"精度: {accuracy:.4f}")
        else:
            print(f"精度: {accuracy}")
        print(f"AutoML有効: {result.get('automl_enabled', 'N/A')}")
        print(f"特徴量重要度あり: {'feature_importance' in result}")

        return result.get("success", False)

    except Exception as e:
        print(f"AutoML + アンサンブル統合テストでエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    print("🚀 アンサンブル学習統合テストを開始")
    print("=" * 70)

    results = []

    # 各テストを実行
    results.append(("MLTrainingService統合", test_ml_training_service_ensemble()))
    results.append(("スタッキング統合", test_stacking_integration()))
    results.append(("AutoML+アンサンブル統合", test_automl_ensemble_integration()))

    # 結果をまとめ
    print("\n" + "=" * 70)
    print("=== 統合テスト結果まとめ ===")
    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 全ての統合テストが成功しました！")
        print("アンサンブル学習がMLTrainingServiceレベルで正しく動作しています。")
        print("LightGBMオンリーからアンサンブル学習への移行が完了しました。")
    else:
        print("\n⚠️ 一部の統合テストが失敗しました。")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
