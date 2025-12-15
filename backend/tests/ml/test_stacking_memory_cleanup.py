import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.ensemble.stacking import StackingEnsemble
from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer


class TestStackingMemoryCleanup:
    """StackingEnsembleとEnsembleTrainerのメモリクリーンアップ機能のテスト"""

    @pytest.fixture
    def mock_data(self):
        """テスト用データ"""
        X = pd.DataFrame(
            {"feature1": np.random.rand(100), "feature2": np.random.rand(100)}
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y

    @pytest.fixture
    def stacking_config(self):
        """StackingEnsembleの設定"""
        return {
            "base_models": ["lightgbm"],
            "meta_model": "logistic_regression",
            "cv_folds": 2,
            "stack_method": "predict_proba",
        }

    def test_stacking_ensemble_clear_training_data(self, mock_data, stacking_config):
        """StackingEnsemble.clear_training_dataが正しく変数をクリアするかテスト"""
        X, y = mock_data

        # モデルの初期化
        model = StackingEnsemble(stacking_config)

        # 内部メソッドをモック化して学習プロセスをバイパス
        with (
            patch.object(model, "_create_base_estimators") as mock_create_estimators,
            patch.object(model, "_create_cv_splitter") as mock_create_cv,
            patch(
                "app.services.ml.ensemble.stacking.cross_val_predict"
            ) as mock_cv_predict,
            patch.object(model, "_create_base_model") as mock_create_meta,
        ):

            # モックの設定
            mock_est = MagicMock()
            mock_est.fit.return_value = mock_est
            mock_est.predict_proba.return_value = np.zeros((len(X), 2))
            # (name, estimator) のリストを返す
            mock_create_estimators.return_value = [("lightgbm", mock_est)]

            # CVの結果 (OOF予測)
            # cross_val_predictは (n_samples, n_classes) または (n_samples,) を返す
            mock_cv_predict.return_value = np.zeros((len(X), 2))

            # メタモデル
            mock_meta = MagicMock()
            mock_meta.fit.return_value = mock_meta
            mock_meta.predict_proba.return_value = np.zeros((len(X), 2))
            mock_create_meta.return_value = mock_meta

            # 実際にfitを実行
            model.fit(X, y)

        # 学習直後はデータが存在することを確認
        assert model.X_train_original is not None
        # DataFrameの比較はpd.testingを使用
        pd.testing.assert_frame_equal(model.X_train_original, X)
        assert model.y_train_original is not None
        assert model.oof_predictions is not None

        # クリーンアップ実行
        model.clear_training_data()

        # データがクリアされたことを確認
        assert model.X_train_original is None
        assert model.y_train_original is None
        assert model.oof_predictions is None
        assert model.oof_base_model_predictions is None

    def test_ensemble_trainer_auto_cleanup(self, mock_data, stacking_config):
        """EnsembleTrainerが学習後に自動的にclear_training_dataを呼ぶかテスト"""
        X, y = mock_data
        X_small = X.iloc[:20]
        y_small = y.iloc[:20]

        # Trainer設定
        ensemble_config = {
            "method": "stacking",
            "stacking_params": stacking_config,
            # メタラベリング有効化（これにより内部フローブランチを通過させる）
            "meta_labeling_params": {
                "model_type": "logistic_regression",  # 軽量モデル
                "model_params": {},
            },
        }

        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        # StackingEnsembleのclear_training_dataが呼ばれたか監視するためのスパイ
        # ただし、trainer内部で初期化されるため、クラスをパッチする必要がある

        with patch(
            "app.services.ml.ensemble.ensemble_trainer.StackingEnsemble"
        ) as MockStackingClass:
            # モックインスタンスの設定
            mock_instance = MockStackingClass.return_value
            mock_instance.fit.return_value = {"status": "success"}  # fitの戻り値
            mock_instance.predict_proba.return_value = np.zeros((20, 2))
            mock_instance.get_oof_predictions.return_value = np.zeros(20)
            mock_instance.get_oof_base_model_predictions.return_value = pd.DataFrame()
            mock_instance.get_X_train_original.return_value = X_small
            mock_instance.get_y_train_original.return_value = y_small

            # 特徴量カラム等の属性設定
            # feature_importanceは辞書である必要がある
            mock_instance.feature_importance = {}
            mock_instance.get_feature_importance.return_value = {}

            # メタラベリングサービスもモック化して学習を成功させる
            with patch(
                "app.services.ml.ensemble.ensemble_trainer.MetaLabelingService"
            ) as MockMetaService:
                mock_meta_instance = MockMetaService.return_value
                mock_meta_instance.train.return_value = {"status": "success"}
                mock_meta_instance.is_trained = True

                # トレーニング実行
                trainer.train_model(
                    X_small, None, None, save_model=False, time_limit=None
                )  # yは内部で生成または不要？

                # BaseMLTrainer.train_model -> ラベル生成 -> _train_model_impl
                # 単体テストとしては _train_model_impl を直接テストするのが早い

                trainer._train_model_impl(
                    X_train=X_small, y_train=y_small, X_test=X_small, y_test=y_small
                )

                # clear_training_dataが呼ばれたことを確認
                mock_instance.clear_training_data.assert_called_once()




