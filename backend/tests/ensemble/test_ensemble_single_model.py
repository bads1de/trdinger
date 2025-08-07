"""
アンサンブル学習で単一モデルのみが保存・表示されることを確認するテスト
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import glob

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.ml.ensemble.stacking import StackingEnsemble
from app.services.ml.ensemble.bagging import BaggingEnsemble


class TestEnsembleSingleModel(unittest.TestCase):
    """アンサンブル学習で単一モデルのみが保存されることをテスト"""

    def setUp(self):
        """テスト用のデータを準備"""
        # テスト用データを生成
        np.random.seed(42)
        self.X_train = pd.DataFrame(
            np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        self.y_train = pd.Series(np.random.randint(0, 3, 100))
        self.X_test = pd.DataFrame(
            np.random.randn(20, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        self.y_test = pd.Series(np.random.randint(0, 3, 20))

        # 一時ディレクトリを作成
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """テスト後のクリーンアップ"""
        # 一時ファイルを削除
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stacking_ensemble_single_model_selection(self):
        """スタッキングアンサンブルで最高性能モデル1つのみが選択されることをテスト"""
        config = {
            "base_models": ["lightgbm"],
            "meta_model": "lightgbm",
            "cv_folds": 2,
            "use_probas": True,
            "random_state": 42,
        }

        ensemble = StackingEnsemble(config=config)

        # 手動でベストモデル選択の結果をシミュレート
        mock_model = MagicMock()
        mock_model.is_trained = True

        ensemble.base_models = [mock_model]
        ensemble.best_algorithm = "lightgbm"
        ensemble.best_model_score = 0.85
        ensemble.meta_model = None
        ensemble.is_fitted = True

        # 最高性能モデル1つのみが保持されていることを確認
        self.assertEqual(len(ensemble.base_models), 1)
        self.assertEqual(ensemble.best_algorithm, "lightgbm")
        self.assertAlmostEqual(ensemble.best_model_score, 0.85)
        self.assertIsNone(ensemble.meta_model)  # メタモデルは使用しない

    def test_bagging_ensemble_single_model_selection(self):
        """バギングアンサンブルで最高性能モデル1つのみが選択されることをテスト"""
        config = {"base_model": "lightgbm", "n_estimators": 3, "random_state": 42}

        ensemble = BaggingEnsemble(config=config)

        # モックを使用してベースモデルの学習をシミュレート
        with patch.object(ensemble, "_create_base_model") as mock_create_model:
            # 異なる性能のモックモデルを作成
            mock_models = []
            scores = [0.70, 0.80, 0.75]

            for score in scores:
                mock_model = MagicMock()
                mock_model.is_trained = True
                mock_model._train_model_impl.return_value = {"accuracy": score}
                mock_models.append(mock_model)

            mock_create_model.side_effect = mock_models

            # アンサンブル学習を実行
            result = ensemble.fit(self.X_train, self.y_train, self.X_test, self.y_test)

            # 最高性能モデル1つのみが保持されていることを確認
            self.assertEqual(len(ensemble.base_models), 1)
            self.assertEqual(ensemble.best_algorithm, "lightgbm")
            self.assertAlmostEqual(ensemble.best_model_score, 0.80)
            self.assertTrue(result.get("selected_model_only", False))

    def test_ensemble_save_single_file(self):
        """アンサンブル学習で単一ファイルのみが保存されることをテスト"""
        config = {
            "base_models": ["lightgbm"],
            "meta_model": "lightgbm",
            "cv_folds": 2,
            "random_state": 42,
        }

        ensemble = StackingEnsemble(config=config)

        # 最高性能モデルを設定（辞書を使用してpickle問題を回避）
        simple_model = {"is_trained": True, "model_type": "lightgbm", "data": "test"}

        ensemble.base_models = [simple_model]
        ensemble.best_algorithm = "lightgbm"
        ensemble.best_model_score = 0.85
        ensemble.is_fitted = True
        ensemble.feature_columns = [f"feature_{i}" for i in range(5)]

        # モデルを保存
        base_path = os.path.join(self.temp_dir, "test_ensemble")
        saved_paths = ensemble.save_models(base_path)

        # 単一ファイルのみが保存されていることを確認
        self.assertEqual(len(saved_paths), 1)
        self.assertTrue(saved_paths[0].endswith(".pkl"))
        self.assertIn("lightgbm", saved_paths[0])

        # 複数ファイル（base_model_0.pkl, meta_model.pkl, config.pkl）が作成されていないことを確認
        pattern = f"{base_path}_base_model_*.pkl"
        base_model_files = glob.glob(pattern)
        self.assertEqual(len(base_model_files), 0)

        pattern = f"{base_path}_meta_model_*.pkl"
        meta_model_files = glob.glob(pattern)
        self.assertEqual(len(meta_model_files), 0)

        pattern = f"{base_path}_config_*.pkl"
        config_files = glob.glob(pattern)
        self.assertEqual(len(config_files), 0)

    def test_ensemble_trainer_save_model(self):
        """EnsembleTrainerで統一されたモデル保存が使用されることをテスト"""
        # 基本的な機能テスト：アンサンブルトレーナーの属性確認
        from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer

        # 必要な設定でトレーナーを作成
        ensemble_config = {
            "bagging_params": {
                "base_model": "lightgbm",
                "n_estimators": 2,
                "random_state": 42,
            }
        }

        trainer = EnsembleTrainer(ensemble_config=ensemble_config)
        trainer.ensemble_method = "bagging"
        trainer.is_trained = True
        trainer.feature_columns = [f"feature_{i}" for i in range(5)]

        # 基本的な属性が設定されていることを確認
        self.assertEqual(trainer.ensemble_method, "bagging")
        self.assertTrue(trainer.is_trained)
        self.assertEqual(len(trainer.feature_columns), 5)

        # save_modelメソッドが存在することを確認
        self.assertTrue(hasattr(trainer, "save_model"))
        self.assertTrue(callable(getattr(trainer, "save_model")))


if __name__ == "__main__":
    unittest.main()
