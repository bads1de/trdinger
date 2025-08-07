"""
KNNモデルの実際のMLトレーニング統合テスト

実際のMLトレーニングフローでKNNモデルが正常に動作することを確認します。
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer


class TestKNNIntegration(unittest.TestCase):
    """KNNモデルの統合テストクラス"""

    def setUp(self):
        """テスト用データの準備"""
        # サンプルデータを作成
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(150, 250, 100),
            'Low': np.random.uniform(50, 150, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000, 10000, 100),
        }, index=dates)
        
        # 特徴量データを作成
        self.features_df = self.sample_data.copy()
        for i in range(5):  # 追加の特徴量
            self.features_df[f'feature_{i}'] = np.random.uniform(-1, 1, 100)
            
        # ターゲットデータを作成
        self.target_data = pd.Series(
            np.random.choice([0, 1, 2], size=100, p=[0.3, 0.4, 0.3]),
            index=dates,
            name='target'
        )

    def test_knn_ensemble_training(self):
        """KNNアンサンブル学習の統合テスト"""
        print("\n=== KNNアンサンブル学習統合テスト ===")
        
        # アンサンブル設定（KNNを使用）
        ensemble_config = {
            "enabled": True,
            "method": "bagging",
            "bagging_params": {
                "n_estimators": 2,
                "bootstrap_fraction": 0.8,
                "base_model_type": "knn",
                "random_state": 42
            }
        }
        
        try:
            # EnsembleTrainerを作成
            ensemble_trainer = EnsembleTrainer(
                ensemble_config=ensemble_config,
                automl_config={}
            )
            
            # 学習データを準備
            training_data = self.features_df.copy()
            training_data['target'] = self.target_data
            
            # 学習実行（テスト用にsave_model=Falseを指定）
            result = ensemble_trainer.train_model(
                training_data=training_data, 
                save_model=False
            )
            
            # 結果の検証
            self.assertIsInstance(result, dict, "学習結果は辞書であるべきです")
            self.assertTrue(ensemble_trainer.is_trained, "学習後はis_trainedがTrueになるべきです")
            self.assertIsNotNone(ensemble_trainer.ensemble_model, "ensemble_modelが設定されていません")
            
            # BaggingClassifierが設定されているかチェック
            self.assertTrue(
                hasattr(ensemble_trainer.ensemble_model, 'bagging_classifier'),
                "bagging_classifierが設定されていません"
            )
            self.assertIsNotNone(
                ensemble_trainer.ensemble_model.bagging_classifier,
                "bagging_classifierがNoneです"
            )
            
            # 学習サンプル数の検証
            self.assertIn('training_samples', result, "training_samplesが結果に含まれていません")
            self.assertGreater(result['training_samples'], 0, "training_samplesが0以下です")
            
            # 精度の検証
            self.assertIn('accuracy', result, "accuracyが結果に含まれていません")
            self.assertGreaterEqual(result['accuracy'], 0.0, "accuracyが負の値です")
            self.assertLessEqual(result['accuracy'], 1.0, "accuracyが1.0を超えています")
            
            print("✅ KNNアンサンブル学習統合テスト成功")
            print(f"   - 学習サンプル数: {result['training_samples']}")
            print(f"   - テストサンプル数: {result.get('test_samples', 'N/A')}")
            print(f"   - 精度: {result.get('accuracy', 'N/A'):.4f}")
            print(f"   - F1スコア: {result.get('f1_score', 'N/A'):.4f}")
            
        except Exception as e:
            self.fail(f"KNNアンサンブル学習統合テストで失敗: {e}")

    def test_knn_model_direct_training(self):
        """KNNモデルの直接学習テスト"""
        print("\n=== KNNモデル直接学習テスト ===")
        
        try:
            from app.services.ml.models.knn_wrapper import KNNModel
            
            # KNNModelを直接作成
            knn_model = KNNModel(n_neighbors=3, weights="uniform")
            
            # 学習データとテストデータに分割
            split_idx = int(len(self.features_df) * 0.8)
            X_train = self.features_df.iloc[:split_idx]
            X_test = self.features_df.iloc[split_idx:]
            y_train = self.target_data.iloc[:split_idx]
            y_test = self.target_data.iloc[split_idx:]
            
            # 学習実行
            fitted_model = knn_model.fit(X_train, y_train)
            
            # 学習結果の検証
            self.assertEqual(fitted_model, knn_model, "fitメソッドはselfを返すべきです")
            self.assertTrue(knn_model.is_trained, "学習後はis_trainedがTrueになるべきです")
            self.assertIsNotNone(knn_model.model, "内部モデルが設定されていません")
            self.assertIsNotNone(knn_model.classes_, "classes_属性が設定されていません")
            
            # 予測テスト
            predictions = knn_model.predict(X_test)
            probabilities = knn_model.predict_proba(X_test)
            
            # 予測結果の検証
            self.assertIsInstance(predictions, np.ndarray, "予測結果はndarrayであるべきです")
            self.assertEqual(len(predictions), len(X_test), "予測結果の長さが正しくありません")
            self.assertIsInstance(probabilities, np.ndarray, "予測確率はndarrayであるべきです")
            self.assertEqual(probabilities.shape[0], len(X_test), "予測確率の行数が正しくありません")
            
            # 予測値が有効な範囲内にあることを確認
            unique_predictions = np.unique(predictions)
            for pred in unique_predictions:
                self.assertIn(pred, knn_model.classes_, f"予測値{pred}がclasses_に含まれていません")
            
            # 確率の合計が1に近いことを確認
            prob_sums = np.sum(probabilities, axis=1)
            for prob_sum in prob_sums:
                self.assertAlmostEqual(prob_sum, 1.0, places=5, msg="確率の合計が1になっていません")
            
            print("✅ KNNモデル直接学習テスト成功")
            print(f"   - 学習サンプル数: {len(X_train)}")
            print(f"   - テストサンプル数: {len(X_test)}")
            print(f"   - 予測クラス数: {len(unique_predictions)}")
            print(f"   - classes_: {knn_model.classes_}")
            
        except Exception as e:
            self.fail(f"KNNモデル直接学習テストで失敗: {e}")


if __name__ == '__main__':
    print("KNNモデルの統合テストを開始します...")
    unittest.main(verbosity=2)
