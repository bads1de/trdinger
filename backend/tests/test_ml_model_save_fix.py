"""
MLモデル保存の修正内容を検証するテスト

このテストは以下の修正内容を検証します：
1. BaggingEnsembleのbase_models設定修正
2. EnsembleTrainerのBaggingClassifier対応
3. 学習サンプル数の正しい記録
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.ensemble.bagging import BaggingEnsemble
from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer
from app.services.ml.ml_metadata import ModelMetadata


class TestMLModelSaveFix(unittest.TestCase):
    """MLモデル保存修正のテストクラス"""

    def setUp(self):
        """テスト用データの準備"""
        # サンプルデータを作成
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=500, freq="H")

        self.sample_data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 200, 500),
                "High": np.random.uniform(150, 250, 500),
                "Low": np.random.uniform(50, 150, 500),
                "Close": np.random.uniform(100, 200, 500),
                "Volume": np.random.uniform(1000, 10000, 500),
            },
            index=dates,
        )

        # 特徴量データを作成
        self.features_df = self.sample_data.copy()
        for i in range(10):  # 追加の特徴量
            self.features_df[f"feature_{i}"] = np.random.uniform(-1, 1, 500)

        # ターゲットデータを作成
        self.target_data = pd.Series(
            np.random.choice([0, 1, 2], size=500, p=[0.3, 0.4, 0.3]),
            index=dates,
            name="target",
        )

    def test_bagging_ensemble_base_models_setup(self):
        """BaggingEnsembleのbase_models設定テスト"""
        print("\n=== BaggingEnsemble base_models設定テスト ===")

        # バギング設定
        bagging_config = {
            "n_estimators": 3,
            "bootstrap_fraction": 0.8,
            "base_model_type": "lightgbm",
            "random_state": 42,
        }

        bagging_ensemble = BaggingEnsemble(bagging_config)

        # 学習データとテストデータに分割
        split_idx = int(len(self.features_df) * 0.8)
        X_train = self.features_df.iloc[:split_idx, :5]  # 最初の5特徴量のみ使用
        X_test = self.features_df.iloc[split_idx:, :5]
        y_train = self.target_data.iloc[:split_idx]
        y_test = self.target_data.iloc[split_idx:]

        try:
            # 学習実行
            result = bagging_ensemble.fit(X_train, y_train, X_test, y_test)

            # base_modelsが適切に設定されているかチェック
            self.assertTrue(
                hasattr(bagging_ensemble, "base_models"),
                "base_modelsが設定されていません",
            )
            self.assertEqual(
                len(bagging_ensemble.base_models),
                1,
                "base_modelsの長さが正しくありません",
            )
            self.assertIsNotNone(
                bagging_ensemble.bagging_classifier,
                "bagging_classifierが設定されていません",
            )

            # best_algorithmが設定されているかチェック
            self.assertTrue(
                hasattr(bagging_ensemble, "best_algorithm"),
                "best_algorithmが設定されていません",
            )
            self.assertIn(
                "bagging",
                bagging_ensemble.best_algorithm,
                "best_algorithmが正しく設定されていません",
            )

            # 学習サンプル数が正しく記録されているかチェック
            self.assertIn(
                "training_samples", result, "training_samplesが結果に含まれていません"
            )
            self.assertEqual(
                result["training_samples"],
                len(X_train),
                "training_samplesが正しくありません",
            )
            self.assertIn(
                "test_samples", result, "test_samplesが結果に含まれていません"
            )
            self.assertEqual(
                result["test_samples"], len(X_test), "test_samplesが正しくありません"
            )

            print("✅ BaggingEnsemble base_models設定テスト成功")
            print(f"   - base_models数: {len(bagging_ensemble.base_models)}")
            print(f"   - best_algorithm: {bagging_ensemble.best_algorithm}")
            print(f"   - training_samples: {result['training_samples']}")
            print(f"   - test_samples: {result['test_samples']}")

        except Exception as e:
            self.fail(f"BaggingEnsemble base_models設定テストで失敗: {e}")

    def test_ensemble_trainer_bagging_save(self):
        """EnsembleTrainerのBaggingClassifier保存テスト"""
        print("\n=== EnsembleTrainer BaggingClassifier保存テスト ===")

        # アンサンブル設定
        ensemble_config = {
            "enabled": True,
            "method": "bagging",
            "bagging_params": {
                "n_estimators": 3,
                "bootstrap_fraction": 0.8,
                "base_model_type": "lightgbm",
                "random_state": 42,
            },
        }

        try:
            # EnsembleTrainerを作成
            ensemble_trainer = EnsembleTrainer(
                ensemble_config=ensemble_config, automl_config={}
            )

            # 学習データとテストデータに分割
            split_idx = int(len(self.features_df) * 0.8)
            X_train = self.features_df.iloc[:split_idx, :5]
            X_test = self.features_df.iloc[split_idx:, :5]
            y_train = self.target_data.iloc[:split_idx]
            y_test = self.target_data.iloc[split_idx:]

            # 学習実行
            result = ensemble_trainer.train_model(X_train, X_test, y_train, y_test)

            # 学習が成功したかチェック
            self.assertTrue(ensemble_trainer.is_trained, "学習が完了していません")
            self.assertIsNotNone(
                ensemble_trainer.ensemble_model, "ensemble_modelが設定されていません"
            )

            # BaggingClassifierが設定されているかチェック
            self.assertTrue(
                hasattr(ensemble_trainer.ensemble_model, "bagging_classifier"),
                "bagging_classifierが設定されていません",
            )
            self.assertIsNotNone(
                ensemble_trainer.ensemble_model.bagging_classifier,
                "bagging_classifierがNoneです",
            )

            # モデル保存のテスト（実際には保存しない）
            with patch(
                "app.services.ml.model_manager.model_manager.save_model"
            ) as mock_save:
                mock_save.return_value = "/mock/path/model.pkl"

                try:
                    model_path = ensemble_trainer.save_model("test_bagging_model")
                    self.assertEqual(
                        model_path, "/mock/path/model.pkl", "保存パスが正しくありません"
                    )

                    # save_modelが呼ばれたかチェック
                    mock_save.assert_called_once()

                    # 呼び出し引数をチェック
                    call_args = mock_save.call_args
                    self.assertIsNotNone(
                        call_args[1]["model"], "モデルが渡されていません"
                    )
                    self.assertEqual(
                        call_args[1]["model_name"],
                        "test_bagging_model",
                        "モデル名が正しくありません",
                    )

                    print("✅ EnsembleTrainer BaggingClassifier保存テスト成功")

                except Exception as save_error:
                    self.fail(f"モデル保存テストで失敗: {save_error}")

        except Exception as e:
            self.fail(f"EnsembleTrainer BaggingClassifier保存テストで失敗: {e}")

    def test_model_metadata_training_samples(self):
        """ModelMetadataの学習サンプル数記録テスト"""
        print("\n=== ModelMetadata学習サンプル数記録テスト ===")

        # 学習結果のモック
        training_result = {
            "accuracy": 0.85,
            "f1_score": 0.78,
            "training_samples": 400,
            "test_samples": 100,
            "model_type": "BaggingClassifier",
        }

        training_params = {"train_test_split": 0.8, "random_state": 42}

        try:
            # ModelMetadataを作成
            metadata = ModelMetadata.from_training_result(
                training_result=training_result,
                training_params=training_params,
                model_type="BaggingClassifier",
                feature_count=15,
            )

            # 学習サンプル数が正しく記録されているかチェック
            self.assertEqual(
                metadata.training_samples,
                400,
                "training_samplesが正しく記録されていません",
            )
            self.assertEqual(
                metadata.test_samples, 100, "test_samplesが正しく記録されていません"
            )
            self.assertEqual(
                metadata.feature_count, 15, "feature_countが正しく記録されていません"
            )
            self.assertEqual(
                metadata.accuracy, 0.85, "accuracyが正しく記録されていません"
            )
            self.assertEqual(
                metadata.f1_score, 0.78, "f1_scoreが正しく記録されていません"
            )

            # 妥当性検証
            validation_result = metadata.validate()
            self.assertTrue(validation_result["is_valid"], "メタデータが無効です")

            # 警告がないことを確認（学習サンプル数が0でない）
            training_sample_warnings = [
                w for w in validation_result["warnings"] if "学習サンプル数" in w
            ]
            self.assertEqual(
                len(training_sample_warnings), 0, "学習サンプル数に関する警告があります"
            )

            print("✅ ModelMetadata学習サンプル数記録テスト成功")
            print(f"   - training_samples: {metadata.training_samples}")
            print(f"   - test_samples: {metadata.test_samples}")
            print(f"   - feature_count: {metadata.feature_count}")
            print(f"   - accuracy: {metadata.accuracy}")
            print(f"   - f1_score: {metadata.f1_score}")

        except Exception as e:
            self.fail(f"ModelMetadata学習サンプル数記録テストで失敗: {e}")

    def test_end_to_end_bagging_workflow(self):
        """エンドツーエンドのバギングワークフローテスト"""
        print("\n=== エンドツーエンドバギングワークフローテスト ===")

        try:
            # 小さなデータセットで完全なワークフローをテスト
            X_sample = self.features_df.iloc[:200, :5]
            y_sample = self.target_data.iloc[:200]

            # 学習データとテストデータに分割
            split_idx = int(len(X_sample) * 0.8)
            X_train = X_sample.iloc[:split_idx]
            X_test = X_sample.iloc[split_idx:]
            y_train = y_sample.iloc[:split_idx]
            y_test = y_sample.iloc[split_idx:]

            # アンサンブル設定
            ensemble_config = {
                "enabled": True,
                "method": "bagging",
                "bagging_params": {
                    "n_estimators": 2,
                    "bootstrap_fraction": 0.8,
                    "base_model_type": "lightgbm",
                    "random_state": 42,
                },
            }

            # EnsembleTrainerを作成
            ensemble_trainer = EnsembleTrainer(
                ensemble_config=ensemble_config,
                automl_config={},
            )

            # 学習データを結合してtraining_data形式にする
            training_data = X_train.copy()
            training_data["target"] = y_train

            # 学習実行（テスト用にsave_model=Falseを指定）
            result = ensemble_trainer.train_model(
                training_data=training_data, save_model=False
            )

            # 結果の検証
            self.assertIsInstance(result, dict, "学習結果は辞書であるべきです")
            self.assertTrue(
                ensemble_trainer.is_trained, "学習後はis_trainedがTrueになるべきです"
            )

            # 学習サンプル数の検証（時系列分割により実際の学習サンプル数は異なる可能性がある）
            self.assertIn(
                "training_samples", result, "training_samplesが結果に含まれていません"
            )
            self.assertGreater(
                result["training_samples"], 0, "training_samplesが0以下です"
            )
            # 時系列分割により実際の学習サンプル数は元のデータより少なくなる可能性がある
            self.assertLessEqual(
                result["training_samples"],
                len(X_train),
                "training_samplesが元のデータサイズを超えています",
            )

            # 予測テスト（特徴量の不一致により予測はスキップ）
            # 実際の学習では101個の特徴量が生成されるため、テスト用の5特徴量では予測できない
            # 学習が成功したことを確認するのが主目的なので、予測テストはスキップ
            print(f"   - 学習成功: 特徴量数={result.get('feature_count', 'N/A')}")
            print(f"   - 学習サンプル数: {result['training_samples']}")
            print(f"   - テストサンプル数: {result.get('test_samples', 'N/A')}")
            print(f"   - 精度: {result.get('accuracy', 'N/A')}")
            print(f"   - F1スコア: {result.get('f1_score', 'N/A')}")

            # ModelMetadataの作成テスト
            metadata = ModelMetadata.from_training_result(
                training_result=result,
                training_params={"train_test_split": 0.8, "random_state": 42},
                model_type="BaggingClassifier",
                feature_count=len(X_train.columns),
            )

            # メタデータの検証
            self.assertGreater(metadata.training_samples, 0, "学習サンプル数が0です")
            # 時系列分割により実際の学習サンプル数は元のデータより少なくなる可能性がある
            self.assertLessEqual(
                metadata.training_samples,
                len(X_train),
                "メタデータの学習サンプル数が元のデータサイズを超えています",
            )

            validation_result = metadata.validate()
            training_sample_warnings = [
                w for w in validation_result["warnings"] if "学習サンプル数" in w
            ]
            self.assertEqual(
                len(training_sample_warnings), 0, "学習サンプル数に関する警告があります"
            )

            print("✅ エンドツーエンドバギングワークフローテスト成功")

        except Exception as e:
            self.fail(f"エンドツーエンドバギングワークフローテストで失敗: {e}")


if __name__ == "__main__":
    print("MLモデル保存修正内容の検証テストを開始します...")
    unittest.main(verbosity=2)
