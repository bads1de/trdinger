"""
StackingEnsembleクラスの基本動作テスト
"""

import os
import sys
import tempfile
import unittest
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.ml.ensemble.stacking import StackingEnsemble


class TestStackingEnsemble(unittest.TestCase):
    """StackingEnsembleクラスのテスト"""

    def setUp(self):
        """テスト用データの準備"""
        # サンプルデータを作成
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        # 特徴量を生成
        X = np.random.randn(n_samples, n_features)
        
        # 3クラス分類のターゲットを生成
        y = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
        
        # DataFrameとSeriesに変換
        feature_names = [f"feature_{i}" for i in range(n_features)]
        self.X_train = pd.DataFrame(X[:150], columns=feature_names)
        self.X_test = pd.DataFrame(X[150:], columns=feature_names)
        self.y_train = pd.Series(y[:150], name="target")
        self.y_test = pd.Series(y[150:], name="target")

    def test_stacking_ensemble_initialization(self):
        """StackingEnsemble初期化テスト"""
        print("\n=== StackingEnsemble初期化テスト ===")
        
        config = {
            "base_models": ["random_forest", "gradient_boosting"],
            "meta_model": "logistic_regression",
            "cv_folds": 3,
            "stack_method": "predict_proba",
            "random_state": 42,
            "n_jobs": 1,
        }
        
        ensemble = StackingEnsemble(config)
        
        self.assertEqual(ensemble.base_models, ["random_forest", "gradient_boosting"])
        self.assertEqual(ensemble.meta_model, "logistic_regression")
        self.assertEqual(ensemble.cv_folds, 3)
        self.assertEqual(ensemble.stack_method, "predict_proba")
        self.assertFalse(ensemble.is_fitted)
        self.assertIsNone(ensemble.stacking_classifier)
        
        print("✅ 初期化テスト完了")

    def test_stacking_ensemble_fit_and_predict(self):
        """StackingEnsemble学習・予測テスト"""
        print("\n=== StackingEnsemble学習・予測テスト ===")
        
        config = {
            "base_models": ["random_forest", "gradient_boosting"],
            "meta_model": "logistic_regression",
            "cv_folds": 2,  # テスト用に小さく設定
            "stack_method": "predict_proba",
            "random_state": 42,
            "n_jobs": 1,
        }
        
        ensemble = StackingEnsemble(config)
        
        # 学習実行
        print("🔄 学習開始...")
        result = ensemble.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        print("✅ 学習完了")
        
        # 学習結果の確認
        self.assertTrue(ensemble.is_fitted)
        self.assertIsNotNone(ensemble.stacking_classifier)
        self.assertEqual(result["model_type"], "StackingClassifier")
        self.assertIn("accuracy", result)
        
        # 予測実行
        print("🔄 予測開始...")
        predictions = ensemble.predict(self.X_test)
        pred_proba = ensemble.predict_proba(self.X_test)
        print("✅ 予測完了")
        
        # 予測結果の確認
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(pred_proba.shape, (len(self.X_test), 3))  # 3クラス分類
        
        print(f"📈 予測結果:")
        print(f"   予測数: {len(predictions)}")
        print(f"   予測確率形状: {pred_proba.shape}")
        print(f"   精度: {result.get('accuracy', 'N/A'):.4f}" if "accuracy" in result else "   精度: N/A")

    def test_stacking_ensemble_save_load(self):
        """StackingEnsembleモデル保存・読み込みテスト"""
        print("\n=== StackingEnsembleモデル保存・読み込みテスト ===")
        
        config = {
            "base_models": ["random_forest"],
            "meta_model": "logistic_regression",
            "cv_folds": 2,
            "stack_method": "predict_proba",
            "random_state": 42,
            "n_jobs": 1,
        }
        
        # 学習
        ensemble1 = StackingEnsemble(config)
        ensemble1.fit(self.X_train, self.y_train)
        
        # 予測（保存前）
        pred1 = ensemble1.predict(self.X_test)
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # モデル保存
            save_success = ensemble1.save_models(model_path)
            self.assertTrue(save_success)
            print("✅ モデル保存完了")
            
            # 新しいインスタンスで読み込み
            ensemble2 = StackingEnsemble(config)
            load_success = ensemble2.load_models(model_path)
            self.assertTrue(load_success)
            self.assertTrue(ensemble2.is_fitted)
            print("✅ モデル読み込み完了")
            
            # 予測（読み込み後）
            pred2 = ensemble2.predict(self.X_test)
            
            # 予測結果が一致することを確認
            np.testing.assert_array_equal(pred1, pred2)
            print("✅ 保存・読み込み後の予測結果が一致")
            
        finally:
            # 一時ファイルを削除
            if os.path.exists(model_path):
                os.unlink(model_path)
            metadata_path = model_path.replace(".pkl", "_metadata.json")
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)

    def test_feature_importance(self):
        """特徴量重要度取得テスト"""
        print("\n=== 特徴量重要度取得テスト ===")
        
        config = {
            "base_models": ["random_forest"],
            "meta_model": "logistic_regression",
            "cv_folds": 2,
            "stack_method": "predict_proba",
            "random_state": 42,
            "n_jobs": 1,
        }
        
        ensemble = StackingEnsemble(config)
        ensemble.fit(self.X_train, self.y_train)
        
        # 特徴量重要度を取得
        importance = ensemble.get_feature_importance()
        
        # 重要度が取得できることを確認
        self.assertIsInstance(importance, dict)
        print(f"✅ 特徴量重要度取得完了: {len(importance)}個の特徴量")

    def test_base_model_predictions(self):
        """ベースモデル予測取得テスト"""
        print("\n=== ベースモデル予測取得テスト ===")
        
        config = {
            "base_models": ["random_forest", "gradient_boosting"],
            "meta_model": "logistic_regression",
            "cv_folds": 2,
            "stack_method": "predict_proba",
            "random_state": 42,
            "n_jobs": 1,
        }
        
        ensemble = StackingEnsemble(config)
        ensemble.fit(self.X_train, self.y_train)
        
        # ベースモデルの予測を取得
        base_predictions = ensemble.get_base_model_predictions(self.X_test)
        
        # 予測が取得できることを確認
        self.assertIsInstance(base_predictions, dict)
        self.assertEqual(len(base_predictions), 2)  # 2つのベースモデル
        print(f"✅ ベースモデル予測取得完了: {list(base_predictions.keys())}")


if __name__ == "__main__":
    print("🚀 StackingEnsembleテスト開始")
    unittest.main(verbosity=2)
