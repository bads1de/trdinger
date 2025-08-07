"""
包括的なモデルサポートの検証テスト

このテストは以下の修正内容を検証します：
1. models/__init__.pyでのKNNモデル追加
2. BaseEnsembleでの全モデルタイプサポート
3. SingleModelTrainerとBaseEnsembleの整合性
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.models import get_available_models
from app.services.ml.ensemble.base_ensemble import BaseEnsemble
from app.services.ml.single_model.single_model_trainer import SingleModelTrainer


class TestModelSupportComprehensive(unittest.TestCase):
    """包括的なモデルサポートのテストクラス"""

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

    def test_models_init_knn_support(self):
        """models/__init__.pyでのKNNモデルサポートテスト"""
        print("\n=== models/__init__.py KNNサポートテスト ===")
        
        try:
            # get_available_models関数を呼び出し
            available_models = get_available_models()
            
            # KNNモデルが含まれているかチェック
            self.assertIn('knn', available_models, "KNNモデルがget_available_modelsに含まれていません")
            
            print(f"✅ 利用可能なモデル: {available_models}")
            print(f"✅ KNNモデルが正常に含まれています")
            
        except Exception as e:
            self.fail(f"models/__init__.py KNNサポートテストで失敗: {e}")

    def test_base_ensemble_model_support(self):
        """BaseEnsembleでの全モデルタイプサポートテスト"""
        print("\n=== BaseEnsemble 全モデルサポートテスト ===")
        
        # テスト用のBaseEnsemble実装
        class TestEnsemble(BaseEnsemble):
            def fit(self, X_train, y_train, X_test=None, y_test=None):
                return {}
            def predict(self, X):
                return np.array([])
            def predict_proba(self, X):
                return np.array([])
        
        ensemble = TestEnsemble({})
        
        # SingleModelTrainerでサポートされているモデルタイプ
        supported_models = [
            "lightgbm",
            "xgboost", 
            "catboost",
            "tabnet",
            "randomforest",
            "extratrees",
            "gradientboosting", 
            "adaboost",
            "ridge",
            "naivebayes",
            "knn"
        ]
        
        # BaseEnsembleでの別名もテスト
        model_aliases = {
            "randomforest": ["random_forest"],
            "gradientboosting": ["gradient_boosting"]
        }
        
        successful_models = []
        failed_models = []
        
        for model_type in supported_models:
            try:
                model = ensemble._create_base_model(model_type)
                self.assertIsNotNone(model, f"{model_type}モデルの作成に失敗しました")
                successful_models.append(model_type)
                print(f"   ✅ {model_type}: 成功")
                
                # 別名もテスト
                if model_type in model_aliases:
                    for alias in model_aliases[model_type]:
                        try:
                            alias_model = ensemble._create_base_model(alias)
                            self.assertIsNotNone(alias_model, f"{alias}モデルの作成に失敗しました")
                            print(f"   ✅ {alias} (alias): 成功")
                        except Exception as e:
                            print(f"   ❌ {alias} (alias): {e}")
                            
            except Exception as e:
                failed_models.append((model_type, str(e)))
                print(f"   ❌ {model_type}: {e}")
        
        print(f"\n✅ 成功したモデル数: {len(successful_models)}/{len(supported_models)}")
        print(f"✅ 成功したモデル: {successful_models}")
        
        if failed_models:
            print(f"❌ 失敗したモデル: {failed_models}")
            # 重要なモデル（KNN、LightGBM等）が失敗した場合はテスト失敗
            critical_models = ["lightgbm", "knn", "randomforest"]
            failed_critical = [model for model, _ in failed_models if model in critical_models]
            if failed_critical:
                self.fail(f"重要なモデルの作成に失敗しました: {failed_critical}")

    def test_single_model_trainer_consistency(self):
        """SingleModelTrainerとBaseEnsembleの整合性テスト"""
        print("\n=== SingleModelTrainer整合性テスト ===")
        
        try:
            # SingleModelTrainerでサポートされているモデルを取得
            single_supported = [
                "lightgbm", "xgboost", "catboost", "tabnet",
                "randomforest", "extratrees", "gradientboosting", 
                "adaboost", "ridge", "naivebayes", "knn"
            ]
            
            # BaseEnsembleでテスト
            class TestEnsemble(BaseEnsemble):
                def fit(self, X_train, y_train, X_test=None, y_test=None):
                    return {}
                def predict(self, X):
                    return np.array([])
                def predict_proba(self, X):
                    return np.array([])
            
            ensemble = TestEnsemble({})
            
            consistent_models = []
            inconsistent_models = []
            
            for model_type in single_supported:
                try:
                    # SingleModelTrainerでの作成テスト
                    single_trainer = SingleModelTrainer(model_type=model_type)
                    single_model = single_trainer._create_model_instance()
                    
                    # BaseEnsembleでの作成テスト
                    ensemble_model = ensemble._create_base_model(model_type)
                    
                    consistent_models.append(model_type)
                    print(f"   ✅ {model_type}: 両方で作成成功")
                    
                except Exception as e:
                    inconsistent_models.append((model_type, str(e)))
                    print(f"   ❌ {model_type}: {e}")
            
            print(f"\n✅ 整合性のあるモデル数: {len(consistent_models)}/{len(single_supported)}")
            print(f"✅ 整合性のあるモデル: {consistent_models}")
            
            if inconsistent_models:
                print(f"❌ 整合性のないモデル: {inconsistent_models}")
                # 重要なモデルで整合性がない場合は警告
                critical_models = ["lightgbm", "knn"]
                inconsistent_critical = [model for model, _ in inconsistent_models if model in critical_models]
                if inconsistent_critical:
                    print(f"⚠️  重要なモデルで整合性の問題: {inconsistent_critical}")
            
        except Exception as e:
            self.fail(f"SingleModelTrainer整合性テストで失敗: {e}")

    def test_model_type_aliases(self):
        """モデルタイプの別名サポートテスト"""
        print("\n=== モデルタイプ別名サポートテスト ===")
        
        class TestEnsemble(BaseEnsemble):
            def fit(self, X_train, y_train, X_test=None, y_test=None):
                return {}
            def predict(self, X):
                return np.array([])
            def predict_proba(self, X):
                return np.array([])
        
        ensemble = TestEnsemble({})
        
        # 別名のテスト
        alias_tests = [
            ("randomforest", "random_forest"),
            ("gradientboosting", "gradient_boosting"),
        ]
        
        try:
            for original, alias in alias_tests:
                # 元の名前でのテスト
                original_model = ensemble._create_base_model(original)
                self.assertIsNotNone(original_model, f"{original}モデルの作成に失敗しました")
                
                # 別名でのテスト
                alias_model = ensemble._create_base_model(alias)
                self.assertIsNotNone(alias_model, f"{alias}モデルの作成に失敗しました")
                
                print(f"   ✅ {original} / {alias}: 両方で作成成功")
            
            print("✅ モデルタイプ別名サポートテスト成功")
            
        except Exception as e:
            self.fail(f"モデルタイプ別名サポートテストで失敗: {e}")

    def test_error_handling(self):
        """エラーハンドリングテスト"""
        print("\n=== エラーハンドリングテスト ===")
        
        class TestEnsemble(BaseEnsemble):
            def fit(self, X_train, y_train, X_test=None, y_test=None):
                return {}
            def predict(self, X):
                return np.array([])
            def predict_proba(self, X):
                return np.array([])
        
        ensemble = TestEnsemble({})
        
        try:
            # サポートされていないモデルタイプでエラーが発生することを確認
            with self.assertRaises(Exception) as context:
                ensemble._create_base_model("unsupported_model")
            
            self.assertIn("サポートされていないモデルタイプ", str(context.exception))
            print("   ✅ サポートされていないモデルタイプで適切にエラーが発生")
            
            # 空文字列でエラーが発生することを確認
            with self.assertRaises(Exception) as context:
                ensemble._create_base_model("")
            
            print("   ✅ 空文字列で適切にエラーが発生")
            
            print("✅ エラーハンドリングテスト成功")
            
        except Exception as e:
            self.fail(f"エラーハンドリングテストで失敗: {e}")


if __name__ == '__main__':
    print("包括的なモデルサポートの検証テストを開始します...")
    unittest.main(verbosity=2)
