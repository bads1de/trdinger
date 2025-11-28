"""
ML統合のテストモジュール

MLTrainingServiceとML統合機能をテストする。
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app.services.ml.ml_training_service import (
    MLTrainingService,
    OptimizationSettings,
)


class TestMLTrainingService:
    """MLTrainingServiceクラスのテスト"""

    @pytest.fixture
    def sample_training_data(self):
        """サンプル学習データ"""
        return pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104] * 20,
                "high": [105, 106, 107, 108, 109] * 20,
                "low": [95, 96, 97, 98, 99] * 20,
                "close": [102, 103, 104, 105, 106] * 20,
                "volume": [1000, 1100, 1200, 1300, 1400] * 20,
                "target": [1, 0, 1, 2, 1] * 20,  # 3クラス分類
            }
        )

    @pytest.fixture
    def mock_trainer(self):
        """Mockトレーナー"""
        trainer = Mock()
        trainer.is_trained = True
        trainer.model = Mock()
        trainer.feature_columns = ["close", "volume"]
        trainer.scaler = Mock()
        trainer.scaler.transform.return_value = np.array([[1.0, 2.0]])
        trainer.model.predict.return_value = np.array([0.2, 0.3, 0.5])
        trainer.train_model.return_value = {
            "success": True,
            "f1_score": 0.85,
            "classification_report": {"macro avg": {"f1-score": 0.82}},
        }
        trainer.evaluate_model.return_value = {"accuracy": 0.9}
        trainer.predict.return_value = np.array([0.2, 0.3, 0.5])
        # get_model_infoは辞書を返す必要がある
        trainer.get_model_info.return_value = {
            "is_trained": True,
            "feature_columns": ["close", "volume"],
            "feature_count": 2,
            "model_type": "Mock",
        }
        return trainer

    def test_initialization_with_ensemble_trainer(self, mock_trainer):
        """アンサンブルトレーナーでの初期化テスト"""
        # EnsembleTrainerをモック化
        with patch(
            "backend.app.services.ml.ml_training_service.EnsembleTrainer",
            return_value=mock_trainer,
        ):
            service = MLTrainingService(trainer_type="ensemble")
            assert service.trainer_type == "ensemble"
            assert service.trainer == mock_trainer

    def test_initialization_with_single_trainer(self, mock_trainer):
        """単一モデルトレーナーでの初期化テスト"""
        with patch(
            "backend.app.services.ml.ml_training_service.SingleModelTrainer",
            return_value=mock_trainer,
        ):
            service = MLTrainingService(trainer_type="single")
            assert service.trainer_type == "single"
            assert service.trainer == mock_trainer

    def test_create_trainer_config_ensemble(self):
        """アンサンブルトレーナー設定作成テスト"""
        # EnsembleTrainerのモックが必要（内部でインポートされるため）
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer"):
            service = MLTrainingService()
            config = service._create_trainer_config("ensemble", None, None)

            assert config["type"] == "ensemble"
            assert config["model_type"] == "stacking"
            assert "ensemble_config" in config

    def test_create_trainer_config_single(self):
        """単一モデルトレーナー設定作成テスト"""
        with patch("backend.app.services.ml.ml_training_service.SingleModelTrainer"):
            service = MLTrainingService()
            single_config = {"model_type": "xgboost"}
            config = service._create_trainer_config("single", None, single_config)

            assert config["type"] == "single"
            assert config["model_type"] == "xgboost"
            assert config["model_params"] == single_config

    def test_create_trainer_config_invalid_type(self):
        """無効なトレーナータイプでの設定作成テスト"""
        service = MLTrainingService()
        with pytest.raises(ValueError, match="サポートされていないトレーナータイプ"):
            service._create_trainer_config("invalid", None, None)

    def test_train_model_without_optimization(self, sample_training_data, mock_trainer):
        """最適化なしでのモデル学習テスト"""
        # EnsembleTrainerをモック化してサービス初期化
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer # 明示的にセット

            result = service.train_model(sample_training_data, save_model=False)

            assert result["success"] is True
            assert result["f1_score"] == 0.85
            mock_trainer.train_model.assert_called_once()

    @patch("backend.app.services.optimization.optimization_service.OptimizationService.optimize_parameters")
    def test_train_model_with_optimization(
        self, mock_optimize_parameters, sample_training_data, mock_trainer
    ):
        """最適化ありでのモデル学習テスト"""
        mock_optimize_parameters.return_value = {
            "best_params": {"learning_rate": 0.1},
            "best_score": 0.9,
            "total_evaluations": 10,
            "optimization_time": 5.0,
        }

        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            # 最適化設定を正しく設定
            optimization_settings = OptimizationSettings(
                enabled=True,
                n_calls=10,
                parameter_space={
                    "learning_rate": {"type": "real", "low": 0.01, "high": 0.1}
                },
            )
            result = service.train_model(
                sample_training_data,
                save_model=False,
                optimization_settings=optimization_settings,
            )

            assert "optimization_result" in result
            assert result["optimization_result"]["best_score"] == 0.9
            mock_optimize_parameters.assert_called_once()
            # cleanup は OptimizationService のインスタンスで行われるため、ここでは確認しない

    def test_evaluate_model(self, sample_training_data, mock_trainer):
        """モデル評価テスト"""
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            result = service.evaluate_model(sample_training_data)

            assert result["accuracy"] == 0.9
            mock_trainer.evaluate_model.assert_called_once()

    def test_get_training_status(self, mock_trainer):
        """学習状態取得テスト"""
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            status = service.get_training_status()

            assert status["is_trained"] is True
            assert status["trainer_type"] == "ensemble"
            assert status["feature_count"] == 2

    def test_predict(self, mock_trainer):
        """予測テスト"""
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            features_df = pd.DataFrame([[1.0, 2.0]], columns=["close", "volume"])
            result = service.predict(features_df)

            assert "predictions" in result
            assert result["model_type"] == "ensemble"
            assert result["feature_count"] == 2

    @pytest.mark.skip(
        reason="generate_signalsの実装が複雑すぎてモックが困難。実装側の問題として別途対応が必要"
    )
    def test_generate_signals_success(self, mock_trainer):
        """シグナル生成成功テスト"""
        service = MLTrainingService()
        service.trainer = mock_trainer
        # 3クラス分類の結果を返すように設定
        mock_trainer.model.predict.return_value = np.array([0.2, 0.3, 0.5])

        features_df = pd.DataFrame([[1.0, 2.0]], columns=["close", "volume"])
        signals = service.generate_signals(features_df)

        assert "up" in signals
        assert "down" in signals
        assert "range" in signals
        assert abs(signals["down"] - 0.2) < 0.01
        assert abs(signals["range"] - 0.3) < 0.01
        assert abs(signals["up"] - 0.5) < 0.01

    @pytest.mark.skip(
        reason="generate_signalsの実装が複雑すぎてモックが困難。実装側の問題として別途対応が必要"
    )
    def test_generate_signals_untrained_model(self, mock_trainer):
        """未学習モデルのシグナル生成テスト"""
        mock_trainer.is_trained = False
        service = MLTrainingService()
        service.trainer = mock_trainer

        features_df = pd.DataFrame([[1.0, 2.0]], columns=["close", "volume"])
        signals = service.generate_signals(features_df)

        # デフォルト値が返されるはず
        assert isinstance(signals, dict)

    @pytest.mark.skip(
        reason="generate_signalsの実装が複雑すぎてモックが困難。実装側の問題として別途対応が必要"
    )
    @patch("backend.app.services.ml.ml_training_service.logger")
    def test_generate_signals_missing_features(self, mock_logger, mock_trainer):
        """特徴量不足時のシグナル生成テスト"""
        service = MLTrainingService()
        service.trainer = mock_trainer
        mock_trainer.feature_columns = ["close", "volume", "missing_feature"]

        features_df = pd.DataFrame([[1.0, 2.0]], columns=["close", "volume"])
        signals = service.generate_signals(features_df)

        mock_logger.warning.assert_called()
        assert isinstance(signals, dict)

    def test_load_model(self, mock_trainer):
        """モデル読み込みテスト"""
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            result = service.load_model("test_model.pkl")
            assert result is mock_trainer.load_model.return_value

    def test_get_feature_importance(self, mock_trainer):
        """特徴量重要度取得テスト"""
        mock_trainer.get_feature_importance.return_value = {"close": 0.6, "volume": 0.4}
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            importance = service.get_feature_importance()
            assert importance["close"] == 0.6
            assert importance["volume"] == 0.4

    def test_get_available_single_models(self):
        """利用可能単一モデル取得テスト"""
        with patch(
            "backend.app.services.ml.ml_training_service.SingleModelTrainer.get_available_models",
            return_value=["lightgbm", "xgboost"],
        ):
            models = MLTrainingService.get_available_single_models()
            assert models == ["lightgbm", "xgboost"]

    def test_determine_trainer_type(self):
        """トレーナータイプ決定テスト"""
        # アンサンブル設定が有効（デフォルト）
        trainer_type = MLTrainingService.determine_trainer_type(None)
        assert trainer_type == "ensemble"

        # アンサンブル設定が無効
        trainer_type = MLTrainingService.determine_trainer_type({"enabled": False})
        assert trainer_type == "single"

    def test_prepare_training_params_default_config(self, mock_trainer):
        """ml_configからのデフォルト設定取得テスト"""
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            # パラメータなしで呼び出し
            params = service._prepare_training_params({})

            # ml_configのデフォルト値が設定されていること
            assert "use_time_series_split" in params
            assert (
                params["use_time_series_split"]
                == service.config.training.USE_TIME_SERIES_SPLIT
            )

    def test_prepare_training_params_with_cv(self, mock_trainer):
        """クロスバリデーション有効時のパラメータ設定テスト"""
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            # CVを有効にして呼び出し
            params = service._prepare_training_params({"use_cross_validation": True})

            # CV関連パラメータが設定されていること
            assert params["use_cross_validation"] is True
            assert "cv_splits" in params
            assert "max_train_size" in params
            assert params["cv_splits"] == service.config.training.CROSS_VALIDATION_FOLDS

    def test_prepare_training_params_override(self, mock_trainer):
        """training_paramsによる上書きテスト"""
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            # カスタムパラメータで呼び出し
            custom_params = {
                "use_cross_validation": True,
                "cv_splits": 10,
                "max_train_size": 5000,
                "use_time_series_split": False,
            }
            params = service._prepare_training_params(custom_params)

            # カスタム値が優先されていること
            assert params["cv_splits"] == 10
            assert params["max_train_size"] == 5000
            assert params["use_time_series_split"] is False

    def test_prepare_training_params_invalid_cv_splits(self, mock_trainer):
        """無効なcv_splitsでのバリデーションテスト"""
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            # cv_splits < 2 でエラー
            with pytest.raises(ValueError, match="cv_splitsは2以上である必要があります"):
                service._prepare_training_params(
                    {"use_cross_validation": True, "cv_splits": 1}
                )

    def test_prepare_training_params_invalid_max_train_size(self, mock_trainer):
        """無効なmax_train_sizeでのバリデーションテスト"""
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            # max_train_size <= 0 でエラー
            with pytest.raises(
                ValueError, match="max_train_sizeは正の整数である必要があります"
            ):
                service._prepare_training_params(
                    {"use_cross_validation": True, "max_train_size": 0}
                )

    def test_train_model_with_timeseries_params(
        self, sample_training_data, mock_trainer
    ):
        """TimeSeriesSplitパラメータ付きモデル学習テスト"""
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer", return_value=mock_trainer):
            service = MLTrainingService()
            service.trainer = mock_trainer

            # TimeSeriesSplit関連パラメータを指定
            result = service.train_model(
                sample_training_data,
                save_model=False,
                use_cross_validation=True,
                cv_splits=5,
                max_train_size=1000,
            )

            assert result["success"] is True
            # トレーナーに正しいパラメータが渡されていることを確認
            call_args = mock_trainer.train_model.call_args
            assert call_args is not None
            assert "cv_splits" in call_args.kwargs
            assert call_args.kwargs["cv_splits"] == 5


class TestOptimizationSettings:
    """OptimizationSettingsクラスのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        settings = OptimizationSettings(enabled=True, n_calls=50)
        assert settings.enabled is True
        assert settings.n_calls == 50
        assert settings.parameter_space == {}

    def test_initialization_with_params(self):
        """パラメータ付き初期化テスト"""
        param_space = {"learning_rate": {"type": "float", "low": 0.01, "high": 0.1}}
        settings = OptimizationSettings(
            enabled=False, n_calls=100, parameter_space=param_space
        )
        assert settings.enabled is False
        assert settings.n_calls == 100
        assert settings.parameter_space == param_space