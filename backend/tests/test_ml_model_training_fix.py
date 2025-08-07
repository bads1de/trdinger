"""
MLモデル学習の修正内容を検証するテスト

このテストは以下の修正内容を検証します：
1. LightGBMModelのsklearn互換性修正
2. データ準備プロセスの問題修正
3. アンサンブル学習のエラーハンドリング強化
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.models.lightgbm_wrapper import LightGBMModel
from app.services.ml.ensemble.bagging import BaggingEnsemble
from app.utils.data_processing import DataProcessor
from app.services.ml.base_ml_trainer import LabelGeneratorWrapper
from sklearn.ensemble import BaggingClassifier


class TestMLModelTrainingFix(unittest.TestCase):
    """MLモデル学習修正のテストクラス"""

    def setUp(self):
        """テスト用データの準備"""
        # サンプルデータを作成
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 1000),
            'High': np.random.uniform(150, 250, 1000),
            'Low': np.random.uniform(50, 150, 1000),
            'Close': np.random.uniform(100, 200, 1000),
            'Volume': np.random.uniform(1000, 10000, 1000),
        }, index=dates)
        
        # 特徴量データを作成
        self.features_df = self.sample_data.copy()
        for i in range(10):  # 追加の特徴量
            self.features_df[f'feature_{i}'] = np.random.uniform(-1, 1, 1000)
            
        # ターゲットデータを作成
        self.target_data = pd.Series(
            np.random.choice([0, 1, 2], size=1000, p=[0.3, 0.4, 0.3]),
            index=dates,
            name='target'
        )

    def test_lightgbm_sklearn_compatibility(self):
        """LightGBMModelのsklearn互換性テスト"""
        print("\n=== LightGBMModel sklearn互換性テスト ===")
        
        # LightGBMModelインスタンスを作成
        model = LightGBMModel()
        
        # fitメソッドが存在することを確認
        self.assertTrue(hasattr(model, 'fit'), "fitメソッドが存在しません")
        self.assertTrue(hasattr(model, 'predict'), "predictメソッドが存在しません")
        self.assertTrue(hasattr(model, 'predict_proba'), "predict_probaメソッドが存在しません")
        
        # 小さなサンプルデータでテスト
        X_sample = self.features_df.iloc[:100, :5]  # 最初の5特徴量のみ使用
        y_sample = self.target_data.iloc[:100]
        
        try:
            # fitメソッドのテスト
            fitted_model = model.fit(X_sample, y_sample)
            self.assertEqual(fitted_model, model, "fitメソッドはselfを返すべきです")
            self.assertTrue(model.is_trained, "学習後はis_trainedがTrueになるべきです")
            
            # predictメソッドのテスト
            predictions = model.predict(X_sample.iloc[:10])
            self.assertIsInstance(predictions, np.ndarray, "予測結果はndarrayであるべきです")
            self.assertEqual(len(predictions), 10, "予測結果の長さが正しくありません")
            
            # predict_probaメソッドのテスト
            probabilities = model.predict_proba(X_sample.iloc[:10])
            self.assertIsInstance(probabilities, np.ndarray, "予測確率はndarrayであるべきです")
            self.assertEqual(probabilities.shape[0], 10, "予測確率の行数が正しくありません")
            
            print("✅ LightGBMModel sklearn互換性テスト成功")
            
        except Exception as e:
            self.fail(f"LightGBMModel sklearn互換性テストで失敗: {e}")

    def test_bagging_classifier_integration(self):
        """BaggingClassifierとLightGBMModelの統合テスト"""
        print("\n=== BaggingClassifier統合テスト ===")
        
        try:
            # LightGBMModelをベースエスティメータとしてBaggingClassifierを作成
            base_model = LightGBMModel()
            bagging_classifier = BaggingClassifier(
                estimator=base_model,
                n_estimators=2,  # テスト用に小さな値
                max_samples=0.8,
                random_state=42
            )
            
            # 小さなサンプルデータでテスト
            X_sample = self.features_df.iloc[:100, :5]
            y_sample = self.target_data.iloc[:100]
            
            # BaggingClassifierの学習テスト
            bagging_classifier.fit(X_sample, y_sample)
            
            # 予測テスト
            predictions = bagging_classifier.predict(X_sample.iloc[:10])
            probabilities = bagging_classifier.predict_proba(X_sample.iloc[:10])
            
            self.assertIsInstance(predictions, np.ndarray, "予測結果はndarrayであるべきです")
            self.assertIsInstance(probabilities, np.ndarray, "予測確率はndarrayであるべきです")
            
            print("✅ BaggingClassifier統合テスト成功")
            
        except Exception as e:
            self.fail(f"BaggingClassifier統合テストで失敗: {e}")

    def test_data_preparation_process(self):
        """データ準備プロセスのテスト"""
        print("\n=== データ準備プロセステスト ===")
        
        try:
            # DataProcessorインスタンスを作成
            data_processor = DataProcessor()
            
            # モックのラベル生成器を作成
            mock_trainer = MagicMock()
            mock_trainer._generate_dynamic_labels.return_value = (
                self.target_data.iloc[:len(self.features_df)], 
                {"threshold_up": 0.02, "threshold_down": -0.02}
            )
            label_generator = LabelGeneratorWrapper(mock_trainer)
            
            # データ準備の実行
            features_clean, labels_clean, threshold_info = data_processor.prepare_training_data(
                self.features_df, 
                label_generator,
                imputation_strategy="median",
                scale_features=True,
                remove_outliers=True
            )
            
            # 結果の検証
            self.assertIsInstance(features_clean, pd.DataFrame, "特徴量はDataFrameであるべきです")
            self.assertIsInstance(labels_clean, pd.Series, "ラベルはSeriesであるべきです")
            self.assertIsInstance(threshold_info, dict, "閾値情報は辞書であるべきです")
            
            self.assertGreater(len(features_clean), 0, "特徴量データが空です")
            self.assertGreater(len(labels_clean), 0, "ラベルデータが空です")
            self.assertEqual(len(features_clean), len(labels_clean), "特徴量とラベルの長さが一致しません")
            
            print(f"✅ データ準備プロセステスト成功: {len(features_clean)}行のデータが準備されました")
            
        except Exception as e:
            self.fail(f"データ準備プロセステストで失敗: {e}")

    def test_bagging_ensemble_error_handling(self):
        """バギングアンサンブルのエラーハンドリングテスト"""
        print("\n=== バギングアンサンブルエラーハンドリングテスト ===")
        
        # バギング設定
        bagging_config = {
            "n_estimators": 2,
            "bootstrap_fraction": 0.8,
            "base_model_type": "lightgbm",
            "random_state": 42
        }
        
        bagging_ensemble = BaggingEnsemble(bagging_config)
        
        # 空データでのエラーテスト
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=int)
        
        with self.assertRaises(Exception) as context:
            bagging_ensemble.fit(empty_df, empty_series)
        
        self.assertIn("空", str(context.exception), "空データエラーメッセージが適切ではありません")
        
        # 長さ不一致データでのエラーテスト
        X_mismatch = self.features_df.iloc[:50, :5]
        y_mismatch = self.target_data.iloc[:100]
        
        with self.assertRaises(Exception) as context:
            bagging_ensemble.fit(X_mismatch, y_mismatch)
        
        self.assertIn("長さ", str(context.exception), "長さ不一致エラーメッセージが適切ではありません")
        
        print("✅ バギングアンサンブルエラーハンドリングテスト成功")

    def test_end_to_end_training_flow(self):
        """エンドツーエンドの学習フローテスト"""
        print("\n=== エンドツーエンド学習フローテスト ===")
        
        try:
            # 小さなデータセットで完全な学習フローをテスト
            X_sample = self.features_df.iloc[:200, :5]
            y_sample = self.target_data.iloc[:200]
            
            # バギングアンサンブルの設定
            bagging_config = {
                "n_estimators": 2,
                "bootstrap_fraction": 0.8,
                "base_model_type": "lightgbm",
                "random_state": 42
            }
            
            # バギングアンサンブルを作成
            bagging_ensemble = BaggingEnsemble(bagging_config)
            
            # 学習データとテストデータに分割
            split_idx = int(len(X_sample) * 0.8)
            X_train = X_sample.iloc[:split_idx]
            X_test = X_sample.iloc[split_idx:]
            y_train = y_sample.iloc[:split_idx]
            y_test = y_sample.iloc[split_idx:]
            
            # 学習実行
            result = bagging_ensemble.fit(X_train, y_train, X_test, y_test)
            
            # 結果の検証
            self.assertIsInstance(result, dict, "学習結果は辞書であるべきです")
            self.assertTrue(bagging_ensemble.is_fitted, "学習後はis_fittedがTrueになるべきです")
            
            # 予測テスト
            predictions = bagging_ensemble.predict(X_test)
            probabilities = bagging_ensemble.predict_proba(X_test)
            
            self.assertEqual(len(predictions), len(X_test), "予測結果の長さが正しくありません")
            self.assertEqual(probabilities.shape[0], len(X_test), "予測確率の行数が正しくありません")
            
            print("✅ エンドツーエンド学習フローテスト成功")
            
        except Exception as e:
            self.fail(f"エンドツーエンド学習フローテストで失敗: {e}")


if __name__ == '__main__':
    print("MLモデル学習修正内容の検証テストを開始します...")
    unittest.main(verbosity=2)
