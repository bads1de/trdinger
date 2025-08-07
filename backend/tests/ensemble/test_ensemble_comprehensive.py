"""
アンサンブル学習の包括的テスト
単一モデル出力の動作を詳細に検証
"""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np
import sys
import glob
import joblib

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.ml.ensemble.stacking import StackingEnsemble
from app.services.ml.ensemble.bagging import BaggingEnsemble
from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer


class TestEnsembleComprehensive(unittest.TestCase):
    """アンサンブル学習の包括的テスト"""

    def setUp(self):
        """テスト用のデータを準備"""
        # テスト用データを生成
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        self.y_train = pd.Series(np.random.randint(0, 3, 100))
        self.X_test = pd.DataFrame(np.random.randn(20, 5), columns=[f'feature_{i}' for i in range(5)])
        self.y_test = pd.Series(np.random.randint(0, 3, 20))
        
        # 一時ディレクトリを作成
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """テスト後のクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bagging_ensemble_real_training(self):
        """バギングアンサンブルの実際の学習テスト"""
        print("\n=== バギングアンサンブル実学習テスト ===")
        
        config = {
            "base_model": "lightgbm",
            "n_estimators": 3,
            "random_state": 42,
            "bootstrap_fraction": 0.8
        }
        
        ensemble = BaggingEnsemble(config=config)
        
        # 実際に学習を実行
        result = ensemble.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        
        print(f"学習結果: {result}")
        print(f"ベースモデル数: {len(ensemble.base_models)}")
        print(f"最高性能アルゴリズム: {getattr(ensemble, 'best_algorithm', 'なし')}")
        print(f"最高性能スコア: {getattr(ensemble, 'best_model_score', 'なし')}")
        
        # 最高性能モデル1つのみが保持されていることを確認
        self.assertEqual(len(ensemble.base_models), 1, "バギングアンサンブルは最高性能モデル1つのみを保持する必要があります")
        self.assertTrue(hasattr(ensemble, 'best_algorithm'), "best_algorithm属性が必要です")
        self.assertTrue(hasattr(ensemble, 'best_model_score'), "best_model_score属性が必要です")
        self.assertTrue(result.get("selected_model_only", False), "selected_model_onlyフラグが必要です")

    def test_stacking_ensemble_best_model_selection(self):
        """スタッキングアンサンブルのベストモデル選択テスト"""
        print("\n=== スタッキングアンサンブル ベストモデル選択テスト ===")
        
        config = {
            "base_models": ["lightgbm", "random_forest"],
            "meta_model": "lightgbm",
            "cv_folds": 2,
            "random_state": 42
        }
        
        ensemble = StackingEnsemble(config=config)
        
        try:
            # 実際に学習を実行
            result = ensemble.fit(self.X_train, self.y_train, self.X_test, self.y_test)
            
            print(f"学習結果: {result}")
            print(f"ベースモデル数: {len(ensemble.base_models)}")
            print(f"最高性能アルゴリズム: {getattr(ensemble, 'best_algorithm', 'なし')}")
            print(f"最高性能スコア: {getattr(ensemble, 'best_model_score', 'なし')}")
            print(f"メタモデル: {ensemble.meta_model}")
            
            # 最高性能モデル1つのみが保持されていることを確認
            self.assertEqual(len(ensemble.base_models), 1, "スタッキングアンサンブルは最高性能モデル1つのみを保持する必要があります")
            self.assertIsNone(ensemble.meta_model, "メタモデルは使用しない")
            self.assertTrue(hasattr(ensemble, 'best_algorithm'), "best_algorithm属性が必要です")
            self.assertTrue(hasattr(ensemble, 'best_model_score'), "best_model_score属性が必要です")
            self.assertTrue(result.get("selected_model_only", False), "selected_model_onlyフラグが必要です")
            
        except Exception as e:
            print(f"スタッキングアンサンブル学習エラー: {e}")
            self.fail(f"スタッキングアンサンブル学習が失敗しました: {e}")

    def test_ensemble_prediction_functionality(self):
        """アンサンブル予測機能のテスト"""
        print("\n=== アンサンブル予測機能テスト ===")
        
        # バギングアンサンブルで予測テスト
        config = {
            "base_model": "lightgbm",
            "n_estimators": 2,
            "random_state": 42
        }
        
        ensemble = BaggingEnsemble(config=config)
        result = ensemble.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        
        # 予測を実行
        predictions = ensemble.predict(self.X_test)
        pred_proba = ensemble.predict_proba(self.X_test)
        
        print(f"予測結果形状: {predictions.shape}")
        print(f"予測確率形状: {pred_proba.shape}")
        print(f"予測値の範囲: {np.min(predictions)} - {np.max(predictions)}")
        
        # 予測結果の妥当性を確認
        self.assertEqual(len(predictions), len(self.X_test), "予測数がテストデータ数と一致する必要があります")
        self.assertEqual(pred_proba.shape[0], len(self.X_test), "予測確率の行数がテストデータ数と一致する必要があります")
        self.assertEqual(pred_proba.shape[1], 3, "3クラス分類の予測確率である必要があります")
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 2), "予測値は0-2の範囲である必要があります")

    def test_ensemble_file_saving_and_loading(self):
        """アンサンブルモデルの保存・読み込みテスト"""
        print("\n=== アンサンブルモデル保存・読み込みテスト ===")
        
        config = {
            "base_model": "lightgbm",
            "n_estimators": 2,
            "random_state": 42
        }
        
        ensemble = BaggingEnsemble(config=config)
        result = ensemble.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        
        # モデルを保存
        base_path = os.path.join(self.temp_dir, "test_ensemble")
        saved_paths = ensemble.save_models(base_path)
        
        print(f"保存されたファイル数: {len(saved_paths)}")
        print(f"保存されたファイル: {saved_paths}")
        
        # 単一ファイルのみが保存されていることを確認
        self.assertEqual(len(saved_paths), 1, "単一ファイルのみが保存される必要があります")
        self.assertTrue(os.path.exists(saved_paths[0]), "保存されたファイルが存在する必要があります")
        
        # ファイル内容を確認
        model_data = joblib.load(saved_paths[0])
        print(f"保存されたデータのキー: {model_data.keys()}")
        
        self.assertIn("model", model_data, "モデルデータが含まれている必要があります")
        self.assertIn("best_algorithm", model_data, "best_algorithmが含まれている必要があります")
        self.assertIn("selected_model_only", model_data, "selected_model_onlyフラグが含まれている必要があります")
        self.assertTrue(model_data["selected_model_only"], "selected_model_onlyがTrueである必要があります")
        
        # 新しいアンサンブルインスタンスで読み込み
        new_ensemble = BaggingEnsemble(config=config)
        load_success = new_ensemble.load_models(base_path)
        
        print(f"読み込み成功: {load_success}")
        print(f"読み込み後のベースモデル数: {len(new_ensemble.base_models)}")
        
        self.assertTrue(load_success, "モデル読み込みが成功する必要があります")
        self.assertEqual(len(new_ensemble.base_models), 1, "読み込み後も単一モデルである必要があります")

    def test_ensemble_trainer_integration(self):
        """EnsembleTrainerの統合テスト"""
        print("\n=== EnsembleTrainer統合テスト ===")
        
        ensemble_config = {
            "bagging_params": {
                "base_model": "lightgbm",
                "n_estimators": 2,
                "random_state": 42
            }
        }
        
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)
        trainer.ensemble_method = "bagging"
        
        print(f"トレーナー初期化完了: method={trainer.ensemble_method}")
        
        # 基本的な属性確認
        self.assertEqual(trainer.ensemble_method, "bagging")
        self.assertTrue(hasattr(trainer, 'save_model'))
        self.assertTrue(callable(getattr(trainer, 'save_model')))
        
        print("EnsembleTrainer統合テスト完了")

    def test_no_multiple_files_created(self):
        """複数ファイルが作成されないことを確認"""
        print("\n=== 複数ファイル作成防止テスト ===")
        
        config = {
            "base_model": "lightgbm",
            "n_estimators": 2,
            "random_state": 42
        }
        
        ensemble = BaggingEnsemble(config=config)
        result = ensemble.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        
        # モデルを保存
        base_path = os.path.join(self.temp_dir, "test_no_multiple")
        saved_paths = ensemble.save_models(base_path)
        
        # 従来の複数ファイル形式が作成されていないことを確認
        base_model_pattern = f"{base_path}_base_model_*.pkl"
        meta_model_pattern = f"{base_path}_meta_model_*.pkl"
        config_pattern = f"{base_path}_config_*.pkl"
        
        base_model_files = glob.glob(base_model_pattern)
        meta_model_files = glob.glob(meta_model_pattern)
        config_files = glob.glob(config_pattern)
        
        print(f"base_model_*.pklファイル数: {len(base_model_files)}")
        print(f"meta_model_*.pklファイル数: {len(meta_model_files)}")
        print(f"config_*.pklファイル数: {len(config_files)}")
        
        self.assertEqual(len(base_model_files), 0, "base_model_*.pklファイルは作成されない")
        self.assertEqual(len(meta_model_files), 0, "meta_model_*.pklファイルは作成されない")
        self.assertEqual(len(config_files), 0, "config_*.pklファイルは作成されない")
        
        print("複数ファイル作成防止テスト完了")


if __name__ == '__main__':
    unittest.main(verbosity=2)
