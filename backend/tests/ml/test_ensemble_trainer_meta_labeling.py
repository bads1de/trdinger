"""
アンサンブルトレーナーとメタラベリングの統合テスト

StackingEnsembleとMetaLabelingServiceの連携を検証します。
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer
from app.services.ml.ensemble.stacking import (
    StackingEnsemble,
)  # StackingEnsembleを明示的にインポート
from app.services.ml.ensemble.meta_labeling import MetaLabelingService


@pytest.fixture
def sample_data():
    """テスト用サンプルデータ"""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(
        np.random.randint(0, 2, n_samples), name="target"
    )  # 0 or 1 for binary meta-labeling

    return X, y


def test_ensemble_trainer_meta_labeling_integration(sample_data):
    """
    アンサンブルトレーナーとメタラベリングの統合テスト
    """
    X, y = sample_data

    # アンサンブル設定
    config = {
        "method": "stacking",
        "stacking_params": {
            "base_models": ["lightgbm", "xgboost"],
            "meta_model": "logistic_regression",
            "cv_folds": 2,
            "stack_method": "predict_proba",
            "n_jobs": 1,
            "passthrough": False,
            "cv_strategy": "kfold",  # ここを追加
        },
    }

    trainer = EnsembleTrainer(config)

    # X_train, X_test, y_train, y_test を準備
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # StackingEnsembleとMetaLabelingServiceをモック
    with (
        patch.object(trainer, "_calculate_features", return_value=X),
        patch.object(
            trainer, "_prepare_training_data", return_value=(X_train, y_train)
        ),
        patch(
            "app.services.ml.ensemble.ensemble_trainer.StackingEnsemble"
        ) as MockStackingEnsemble,  # StackingEnsembleをモック
        patch.object(
            MetaLabelingService, "train", return_value={"status": "success"}
        ) as mock_meta_train,
    ):
        mock_stacking_ensemble_instance = MockStackingEnsemble.return_value
        # X_testのサイズは40 (200 * 0.2) なので、それに合わせた戻り値を設定
        mock_stacking_ensemble_instance.predict_proba.return_value = np.array(
            [[0.1, 0.9]] * len(X_test)
        )
        mock_stacking_ensemble_instance.predict.return_value = np.array(
            [1] * len(X_test)
        )
        mock_stacking_ensemble_instance.is_fitted = True
        mock_stacking_ensemble_instance.get_oof_predictions.return_value = np.array(
            [0.5] * len(X_train)
        )
        mock_stacking_ensemble_instance.get_oof_base_model_predictions.return_value = (
            pd.DataFrame(np.random.rand(len(X_train), 2), index=X_train.index)
        )
        mock_stacking_ensemble_instance.get_X_train_original.return_value = X_train
        mock_stacking_ensemble_instance.get_y_train_original.return_value = y_train
        mock_stacking_ensemble_instance.fit.return_value = {"success": True}

        # StackingEnsembleのインスタンスがtrainer.ensemble_modelに代入されるようにする
        trainer.ensemble_model = mock_stacking_ensemble_instance

        result = trainer._train_model_impl(
            X_train, X_test, y_train, y_test, random_state=42
        )

        assert result["success"] is True
        assert "meta_model_path" not in result  # save_modelではないのでパスは含まれない
        mock_meta_train.assert_called_once()  # メタラベリングが呼ばれたことを確認


class TestEnsembleTrainerErrorHandling:
    """EnsembleTrainer のエラーハンドリングテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        X = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y = pd.Series(np.random.randint(0, 2, n_samples), name="target")
        return X, y

    @pytest.fixture
    def trained_trainer(self, sample_data):
        """学習済みトレーナーのフィクスチャ"""
        X, y = sample_data
        config = {
            "method": "stacking",
            "models": ["lightgbm"],
            "stacking_params": {
                "base_models": ["lightgbm"],
                "meta_model": "logistic_regression",
                "cv_folds": 2,
                "n_jobs": 1,
            },
        }
        trainer = EnsembleTrainer(config)

        # モックのStackingEnsembleを設定
        mock_ensemble = MagicMock(spec=StackingEnsemble)
        mock_ensemble.is_fitted = True
        mock_ensemble.predict_proba.return_value = np.array([[0.3, 0.7]] * len(X))
        trainer.ensemble_model = mock_ensemble
        trainer.is_trained = True

        return trainer, X

    def test_predict_raises_error_when_base_model_fails_strict_mode(
        self, trained_trainer
    ):
        """
        ベースモデル予測失敗時にエラーが発生することを確認（厳格モード）

        サイレント失敗ではなく、明示的にエラーを発生させるべき。
        """
        trainer, X = trained_trainer

        # メタラベリングサービスをモック
        mock_meta_service = MagicMock(spec=MetaLabelingService)
        mock_meta_service.is_trained = True
        trainer.meta_labeling_service = mock_meta_service

        # predict_base_models_proba が例外を発生させる設定
        trainer.ensemble_model.predict_base_models_proba.side_effect = Exception(
            "Base model prediction failed"
        )

        # strict_mode=True（デフォルト）の場合、エラーが発生するべき
        from app.utils.error_handler import ModelError

        with pytest.raises(ModelError) as exc_info:
            trainer.predict(X)

        assert "ベースモデル予測" in str(exc_info.value)

    def test_predict_returns_no_trade_when_base_model_fails_lenient_mode(
        self, trained_trainer
    ):
        """
        ベースモデル予測失敗時にNo Trade（0）を返すことを確認（寛容モード）

        strict_mode=False の場合、エラーではなく安全なデフォルト値を返す。
        """
        trainer, X = trained_trainer
        trainer.strict_error_mode = False  # 寛容モードに設定

        # メタラベリングサービスをモック
        mock_meta_service = MagicMock(spec=MetaLabelingService)
        mock_meta_service.is_trained = True
        trainer.meta_labeling_service = mock_meta_service

        # predict_base_models_proba が例外を発生させる設定
        trainer.ensemble_model.predict_base_models_proba.side_effect = Exception(
            "Base model prediction failed"
        )

        # 寛容モードの場合、全て0（No Trade）を返すべき
        result = trainer.predict(X)

        assert len(result) == len(X)
        assert np.all(result == 0)  # 全てNo Trade


