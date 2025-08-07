"""
アンサンブル学習の統合テスト
実際のデータを使用した動作確認
"""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.ml.ensemble.bagging import BaggingEnsemble
from app.services.ml.ensemble.stacking import StackingEnsemble


class TestEnsembleIntegration(unittest.TestCase):
    """アンサンブル学習の統合テスト"""

    def setUp(self):
        """テスト用のデータを準備"""
        # より現実的なテストデータを生成
        np.random.seed(42)

        # 特徴量データ（価格、ボリューム、テクニカル指標を模擬）
        n_samples = 200
        n_features = 10

        # 価格関連特徴量（トレンド性を持たせる）
        price_trend = np.cumsum(np.random.randn(n_samples) * 0.01)
        volume_data = np.random.exponential(1000, n_samples)

        # テクニカル指標を模擬
        features = []
        for i in range(n_features):
            if i < 3:  # 価格関連
                feature = price_trend + np.random.randn(n_samples) * 0.1
            elif i < 6:  # ボリューム関連
                feature = np.log(volume_data) + np.random.randn(n_samples) * 0.2
            else:  # その他のテクニカル指標
                feature = np.random.randn(n_samples)
            features.append(feature)

        self.X_train = pd.DataFrame(
            np.column_stack(features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # ターゲット変数（価格変動方向）
        # 3クラス分類を確実にするため、バランスの取れたラベルを生成
        labels = []
        for i in range(n_samples):
            if i % 3 == 0:
                labels.append(0)  # DOWN
            elif i % 3 == 1:
                labels.append(1)  # RANGE
            else:
                labels.append(2)  # UP

        # ランダムにシャッフルして現実的にする
        np.random.shuffle(labels)
        self.y_train = pd.Series(labels)

        # テストデータ
        test_size = 50
        self.X_test = self.X_train.tail(test_size).copy()
        self.y_test = self.y_train.tail(test_size).copy()

        # 学習データからテストデータを除外
        self.X_train = self.X_train.head(n_samples - test_size)
        self.y_train = self.y_train.head(n_samples - test_size)

        # 一時ディレクトリを作成
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """テスト後のクリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bagging_ensemble_full_workflow(self):
        """バギングアンサンブルの完全なワークフローテスト"""
        print("\n=== バギングアンサンブル 完全ワークフローテスト ===")

        config = {
            "base_model_type": "lightgbm",
            "n_estimators": 3,
            "bootstrap_fraction": 0.8,
            "random_state": 42,
        }

        ensemble = BaggingEnsemble(config=config)

        print(f"データサイズ: 学習={len(self.X_train)}, テスト={len(self.X_test)}")
        print(f"特徴量数: {len(self.X_train.columns)}")
        print(f"クラス分布: {self.y_train.value_counts().to_dict()}")

        # 学習実行
        result = ensemble.fit(self.X_train, self.y_train, self.X_test, self.y_test)

        print(f"学習結果:")
        print(f"  - 最高性能アルゴリズム: {result.get('best_algorithm', 'なし')}")
        print(f"  - 最高性能スコア: {result.get('best_model_score', 'なし')}")
        print(f"  - 単一モデル選択: {result.get('selected_model_only', False)}")
        print(f"  - 学習済みモデル数: {len(ensemble.base_models)}")

        # 予測実行
        predictions = ensemble.predict(self.X_test)
        pred_proba = ensemble.predict_proba(self.X_test)

        print(f"予測結果:")
        print(f"  - 予測数: {len(predictions)}")
        print(f"  - 予測値の範囲: {np.min(predictions)} - {np.max(predictions)}")
        print(f"  - 予測確率形状: {pred_proba.shape}")
        print(f"  - 予測分布: {np.bincount(predictions.astype(int))}")

        # 検証
        self.assertEqual(
            len(ensemble.base_models), 1, "最高性能モデル1つのみが保持されている"
        )
        self.assertTrue(
            result.get("selected_model_only", False),
            "単一モデル選択フラグが設定されている",
        )
        self.assertEqual(
            len(predictions), len(self.X_test), "予測数がテストデータ数と一致"
        )
        # 予測確率の形状を確認（2クラスまたは3クラス分類）
        expected_classes = len(np.unique(self.y_train))
        self.assertEqual(
            pred_proba.shape,
            (len(self.X_test), expected_classes),
            f"予測確率が{expected_classes}クラス分類",
        )

        # モデル保存・読み込みテスト
        base_path = os.path.join(self.temp_dir, "bagging_model")
        saved_paths = ensemble.save_models(base_path)

        print(f"保存結果:")
        print(f"  - 保存ファイル数: {len(saved_paths)}")
        print(f"  - 保存パス: {saved_paths[0] if saved_paths else 'なし'}")

        # 新しいインスタンスで読み込み
        new_ensemble = BaggingEnsemble(config=config)
        load_success = new_ensemble.load_models(base_path)

        print(f"読み込み結果:")
        print(f"  - 読み込み成功: {load_success}")
        print(f"  - 読み込み後モデル数: {len(new_ensemble.base_models)}")

        # 読み込み後の予測テスト
        if load_success:
            new_predictions = new_ensemble.predict(self.X_test)
            print(
                f"  - 読み込み後予測一致: {np.array_equal(predictions, new_predictions)}"
            )
            self.assertTrue(
                np.array_equal(predictions, new_predictions), "読み込み後の予測が一致"
            )

        self.assertEqual(len(saved_paths), 1, "単一ファイルのみが保存されている")
        self.assertTrue(load_success, "モデル読み込みが成功")

        print("バギングアンサンブル 完全ワークフローテスト完了")

    def test_stacking_ensemble_best_model_selection(self):
        """スタッキングアンサンブルのベストモデル選択テスト"""
        print("\n=== スタッキングアンサンブル ベストモデル選択テスト ===")

        config = {
            "base_models": ["lightgbm", "random_forest"],
            "meta_model": "lightgbm",
            "cv_folds": 2,
            "use_probas": True,
            "random_state": 42,
        }

        ensemble = StackingEnsemble(config=config)

        print(f"設定:")
        print(f"  - ベースモデル: {config['base_models']}")
        print(f"  - メタモデル: {config['meta_model']}")
        print(f"  - CVフォールド: {config['cv_folds']}")

        # 学習実行
        result = ensemble.fit(self.X_train, self.y_train, self.X_test, self.y_test)

        print(f"学習結果:")
        print(f"  - 最高性能アルゴリズム: {result.get('best_algorithm', 'なし')}")
        print(f"  - 最高性能スコア: {result.get('best_model_score', 'なし')}")
        print(f"  - 単一モデル選択: {result.get('selected_model_only', False)}")
        print(f"  - 学習済みモデル数: {len(ensemble.base_models)}")
        print(f"  - メタモデル: {ensemble.meta_model}")

        # 予測実行
        predictions = ensemble.predict(self.X_test)
        pred_proba = ensemble.predict_proba(self.X_test)

        print(f"予測結果:")
        print(f"  - 予測数: {len(predictions)}")
        print(f"  - 予測確率形状: {pred_proba.shape}")

        # 検証
        self.assertEqual(
            len(ensemble.base_models), 1, "最高性能モデル1つのみが保持されている"
        )
        self.assertIsNone(ensemble.meta_model, "メタモデルは使用されていない")
        self.assertTrue(
            result.get("selected_model_only", False),
            "単一モデル選択フラグが設定されている",
        )
        self.assertIn(
            result.get("best_algorithm"),
            ["lightgbm", "random_forest"],
            "有効なアルゴリズムが選択されている",
        )

        print("スタッキングアンサンブル ベストモデル選択テスト完了")

    def test_ensemble_performance_comparison(self):
        """アンサンブル手法の性能比較テスト"""
        print("\n=== アンサンブル手法性能比較テスト ===")

        # バギングアンサンブル
        bagging_config = {
            "base_model_type": "lightgbm",
            "n_estimators": 2,
            "bootstrap_fraction": 0.8,
            "random_state": 42,
        }

        bagging_ensemble = BaggingEnsemble(config=bagging_config)
        bagging_result = bagging_ensemble.fit(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        # スタッキングアンサンブル
        stacking_config = {
            "base_models": ["lightgbm", "random_forest"],
            "meta_model": "lightgbm",
            "cv_folds": 2,
            "random_state": 42,
        }

        stacking_ensemble = StackingEnsemble(config=stacking_config)
        stacking_result = stacking_ensemble.fit(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        print(f"性能比較:")
        print(f"  バギング:")
        print(f"    - アルゴリズム: {bagging_result.get('best_algorithm', 'なし')}")
        print(f"    - スコア: {bagging_result.get('best_model_score', 'なし')}")
        print(f"    - 精度: {bagging_result.get('accuracy', 'なし')}")

        print(f"  スタッキング:")
        print(f"    - アルゴリズム: {stacking_result.get('best_algorithm', 'なし')}")
        print(f"    - スコア: {stacking_result.get('best_model_score', 'なし')}")
        print(f"    - 精度: {stacking_result.get('accuracy', 'なし')}")

        # 両方とも単一モデル選択が機能していることを確認
        self.assertEqual(
            len(bagging_ensemble.base_models), 1, "バギング: 単一モデル選択"
        )
        self.assertEqual(
            len(stacking_ensemble.base_models), 1, "スタッキング: 単一モデル選択"
        )
        self.assertIsNone(
            stacking_ensemble.meta_model, "スタッキング: メタモデル不使用"
        )

        print("アンサンブル手法性能比較テスト完了")


if __name__ == "__main__":
    unittest.main(verbosity=2)
