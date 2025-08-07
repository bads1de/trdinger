"""
KNNモデルサポートの検証テスト

このテストは以下の修正内容を検証します：
1. BaseEnsembleでのKNNモデルサポート追加
2. KNNModelのsklearn互換性修正
3. BaggingEnsembleでのKNNモデル使用
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.models.knn_wrapper import KNNModel
from app.services.ml.ensemble.bagging import BaggingEnsemble
from app.services.ml.ensemble.base_ensemble import BaseEnsemble
from sklearn.ensemble import BaggingClassifier


class TestKNNModelSupport(unittest.TestCase):
    """KNNモデルサポートのテストクラス"""

    def setUp(self):
        """テスト用データの準備"""
        # サンプルデータを作成
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='h')
        
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 200),
            'High': np.random.uniform(150, 250, 200),
            'Low': np.random.uniform(50, 150, 200),
            'Close': np.random.uniform(100, 200, 200),
            'Volume': np.random.uniform(1000, 10000, 200),
        }, index=dates)
        
        # 特徴量データを作成
        self.features_df = self.sample_data.copy()
        for i in range(5):  # 追加の特徴量
            self.features_df[f'feature_{i}'] = np.random.uniform(-1, 1, 200)
            
        # ターゲットデータを作成
        self.target_data = pd.Series(
            np.random.choice([0, 1, 2], size=200, p=[0.3, 0.4, 0.3]),
            index=dates,
            name='target'
        )

    def test_knn_model_sklearn_compatibility(self):
        """KNNModelのsklearn互換性テスト"""
        print("\n=== KNNModel sklearn互換性テスト ===")
        
        # KNNModelインスタンスを作成
        model = KNNModel()
        
        # fitメソッドが存在することを確認
        self.assertTrue(hasattr(model, 'fit'), "fitメソッドが存在しません")
        self.assertTrue(hasattr(model, 'predict'), "predictメソッドが存在しません")
        self.assertTrue(hasattr(model, 'predict_proba'), "predict_probaメソッドが存在しません")
        self.assertTrue(hasattr(model, 'get_params'), "get_paramsメソッドが存在しません")
        self.assertTrue(hasattr(model, 'set_params'), "set_paramsメソッドが存在しません")
        
        # 小さなサンプルデータでテスト
        X_sample = self.features_df.iloc[:100, :5]  # 最初の5特徴量のみ使用
        y_sample = self.target_data.iloc[:100]
        
        try:
            # fitメソッドのテスト
            fitted_model = model.fit(X_sample, y_sample)
            self.assertEqual(fitted_model, model, "fitメソッドはselfを返すべきです")
            self.assertTrue(model.is_trained, "学習後はis_trainedがTrueになるべきです")
            self.assertIsNotNone(model.classes_, "classes_属性が設定されていません")
            
            # predictメソッドのテスト
            predictions = model.predict(X_sample.iloc[:10])
            self.assertIsInstance(predictions, np.ndarray, "予測結果はndarrayであるべきです")
            self.assertEqual(len(predictions), 10, "予測結果の長さが正しくありません")
            
            # predict_probaメソッドのテスト
            probabilities = model.predict_proba(X_sample.iloc[:10])
            self.assertIsInstance(probabilities, np.ndarray, "予測確率はndarrayであるべきです")
            self.assertEqual(probabilities.shape[0], 10, "予測確率の行数が正しくありません")
            
            # get_paramsメソッドのテスト
            params = model.get_params()
            self.assertIsInstance(params, dict, "パラメータは辞書であるべきです")
            self.assertIn('n_neighbors', params, "n_neighborsパラメータが含まれていません")
            
            # set_paramsメソッドのテスト
            new_model = model.set_params(n_neighbors=7)
            self.assertEqual(new_model, model, "set_paramsメソッドはselfを返すべきです")
            self.assertEqual(model.n_neighbors, 7, "パラメータが正しく設定されていません")
            
            print("✅ KNNModel sklearn互換性テスト成功")
            
        except Exception as e:
            self.fail(f"KNNModel sklearn互換性テストで失敗: {e}")

    def test_base_ensemble_knn_support(self):
        """BaseEnsembleでのKNNモデルサポートテスト"""
        print("\n=== BaseEnsemble KNNモデルサポートテスト ===")
        
        try:
            # BaseEnsembleのインスタンスを作成（テスト用）
            class TestEnsemble(BaseEnsemble):
                def fit(self, X_train, y_train, X_test=None, y_test=None):
                    return {}
                def predict(self, X):
                    return np.array([])
                def predict_proba(self, X):
                    return np.array([])
            
            ensemble = TestEnsemble({})
            
            # KNNモデルの作成テスト
            knn_model = ensemble._create_base_model("knn")
            
            # 作成されたモデルの検証
            self.assertIsInstance(knn_model, KNNModel, "KNNModelのインスタンスが作成されていません")
            self.assertTrue(hasattr(knn_model, 'fit'), "fitメソッドが存在しません")
            self.assertTrue(hasattr(knn_model, 'predict'), "predictメソッドが存在しません")
            self.assertTrue(hasattr(knn_model, 'predict_proba'), "predict_probaメソッドが存在しません")
            
            print("✅ BaseEnsemble KNNモデルサポートテスト成功")
            
        except Exception as e:
            self.fail(f"BaseEnsemble KNNモデルサポートテストで失敗: {e}")

    def test_bagging_classifier_knn_integration(self):
        """BaggingClassifierとKNNModelの統合テスト"""
        print("\n=== BaggingClassifier KNN統合テスト ===")
        
        try:
            # KNNModelをベースエスティメータとしてBaggingClassifierを作成
            base_model = KNNModel()
            bagging_classifier = BaggingClassifier(
                estimator=base_model,
                n_estimators=2,  # テスト用に小さな値
                max_samples=0.8,
                random_state=42
            )
            
            # 小さなサンプルデータでテスト
            X_sample = self.features_df.iloc[:50, :5]
            y_sample = self.target_data.iloc[:50]
            
            # BaggingClassifierの学習テスト
            bagging_classifier.fit(X_sample, y_sample)
            
            # 予測テスト
            predictions = bagging_classifier.predict(X_sample.iloc[:10])
            probabilities = bagging_classifier.predict_proba(X_sample.iloc[:10])
            
            self.assertIsInstance(predictions, np.ndarray, "予測結果はndarrayであるべきです")
            self.assertIsInstance(probabilities, np.ndarray, "予測確率はndarrayであるべきです")
            
            print("✅ BaggingClassifier KNN統合テスト成功")
            
        except Exception as e:
            self.fail(f"BaggingClassifier KNN統合テストで失敗: {e}")

    def test_bagging_ensemble_knn_workflow(self):
        """BaggingEnsembleでのKNNワークフローテスト"""
        print("\n=== BaggingEnsemble KNNワークフローテスト ===")
        
        # バギング設定（KNNを使用）
        bagging_config = {
            "n_estimators": 2,
            "bootstrap_fraction": 0.8,
            "base_model_type": "knn",
            "random_state": 42
        }
        
        try:
            # BaggingEnsembleを作成
            bagging_ensemble = BaggingEnsemble(bagging_config)
            
            # 学習データとテストデータに分割
            split_idx = int(len(self.features_df) * 0.8)
            X_train = self.features_df.iloc[:split_idx, :5]
            X_test = self.features_df.iloc[split_idx:, :5]
            y_train = self.target_data.iloc[:split_idx]
            y_test = self.target_data.iloc[split_idx:]
            
            # 学習実行
            result = bagging_ensemble.fit(X_train, y_train, X_test, y_test)
            
            # 結果の検証
            self.assertIsInstance(result, dict, "学習結果は辞書であるべきです")
            self.assertTrue(bagging_ensemble.is_fitted, "学習後はis_fittedがTrueになるべきです")
            
            # base_modelsが適切に設定されているかチェック
            self.assertTrue(hasattr(bagging_ensemble, 'base_models'), "base_modelsが設定されていません")
            self.assertEqual(len(bagging_ensemble.base_models), 1, "base_modelsの長さが正しくありません")
            self.assertIsNotNone(bagging_ensemble.bagging_classifier, "bagging_classifierが設定されていません")
            
            # best_algorithmが設定されているかチェック
            self.assertTrue(hasattr(bagging_ensemble, 'best_algorithm'), "best_algorithmが設定されていません")
            self.assertIn("bagging_knn", bagging_ensemble.best_algorithm, "best_algorithmが正しく設定されていません")
            
            # 学習サンプル数が正しく記録されているかチェック
            self.assertIn('training_samples', result, "training_samplesが結果に含まれていません")
            self.assertEqual(result['training_samples'], len(X_train), "training_samplesが正しくありません")
            self.assertIn('test_samples', result, "test_samplesが結果に含まれていません")
            self.assertEqual(result['test_samples'], len(X_test), "test_samplesが正しくありません")
            
            # 予測テスト
            predictions = bagging_ensemble.predict(X_test)
            probabilities = bagging_ensemble.predict_proba(X_test)
            
            self.assertEqual(len(predictions), len(X_test), "予測結果の長さが正しくありません")
            self.assertEqual(probabilities.shape[0], len(X_test), "予測確率の行数が正しくありません")
            
            print("✅ BaggingEnsemble KNNワークフローテスト成功")
            print(f"   - base_models数: {len(bagging_ensemble.base_models)}")
            print(f"   - best_algorithm: {bagging_ensemble.best_algorithm}")
            print(f"   - training_samples: {result['training_samples']}")
            print(f"   - test_samples: {result['test_samples']}")
            
        except Exception as e:
            self.fail(f"BaggingEnsemble KNNワークフローテストで失敗: {e}")

    def test_knn_model_parameters(self):
        """KNNモデルのパラメータテスト"""
        print("\n=== KNNモデルパラメータテスト ===")
        
        try:
            # カスタムパラメータでKNNModelを作成
            custom_params = {
                "n_neighbors": 7,
                "weights": "uniform",
                "algorithm": "ball_tree",
                "metric": "manhattan",
                "p": 1
            }
            
            model = KNNModel(**custom_params)
            
            # パラメータが正しく設定されているかチェック
            self.assertEqual(model.n_neighbors, 7, "n_neighborsが正しく設定されていません")
            self.assertEqual(model.weights, "uniform", "weightsが正しく設定されていません")
            self.assertEqual(model.algorithm, "ball_tree", "algorithmが正しく設定されていません")
            self.assertEqual(model.metric, "manhattan", "metricが正しく設定されていません")
            self.assertEqual(model.p, 1, "pが正しく設定されていません")
            
            # get_paramsでパラメータを取得
            params = model.get_params()
            for key, value in custom_params.items():
                self.assertEqual(params[key], value, f"{key}パラメータが正しく取得されていません")
            
            # set_paramsでパラメータを変更
            model.set_params(n_neighbors=10, weights="distance")
            self.assertEqual(model.n_neighbors, 10, "set_paramsでn_neighborsが正しく設定されていません")
            self.assertEqual(model.weights, "distance", "set_paramsでweightsが正しく設定されていません")
            
            print("✅ KNNモデルパラメータテスト成功")
            
        except Exception as e:
            self.fail(f"KNNモデルパラメータテストで失敗: {e}")


if __name__ == '__main__':
    print("KNNモデルサポートの検証テストを開始します...")
    unittest.main(verbosity=2)
