"""
予測ロジックリファクタリングのテスト
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from backend.app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer
from backend.app.services.ml.ml_training_service import MLTrainingService


class TestPredictionLogicRefactoring:
    """予測ロジックのリファクタリングをテスト"""

    @pytest.fixture
    def sample_features(self):
        """テスト用特徴量DataFrame"""
        return pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "feature2": np.random.randn(10),
                "feature3": np.random.randn(10),
            }
        )

    @pytest.fixture
    def ml_service(self):
        """MLTrainingServiceのモック"""
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer"):
            service = MLTrainingService(
                trainer_type="single", single_model_config={"model_type": "lightgbm"}
            )
            # 単一モデルモードでもEnsembleTrainerを使用
            service.trainer = MagicMock(spec=EnsembleTrainer)
            service.trainer.is_trained = True
            service.trainer.is_single_model = True
            return service

    def test_predict_returns_probabilities(self, ml_service, sample_features):
        """predictメソッドが確率を返すことをテスト"""
        # モックの設定
        mock_proba = np.array([[0.1, 0.7, 0.2]] * len(sample_features))
        ml_service.trainer.predict.return_value = mock_proba

        # 予測実行
        result = ml_service.predict(sample_features)

        # 確率が返されることを確認
        assert "predictions" in result
        predictions = result["predictions"]
        assert predictions.shape == (len(sample_features), 3)
        ml_service.trainer.predict.assert_called_once()

    def test_predict_signal_returns_dict(self, ml_service, sample_features):
        """predict_signalメソッドが辞書を返すことをテスト"""
        # モックの設定
        expected_signal = {"up": 0.7, "down": 0.1, "range": 0.2}
        ml_service.trainer.predict_signal.return_value = expected_signal

        # シグナル生成
        signal = ml_service.generate_signals(sample_features)

        # 辞書形式で返されることを確認
        assert isinstance(signal, dict)
        assert "up" in signal
        assert "down" in signal
        assert "range" in signal
        ml_service.trainer.predict_signal.assert_called_once()

    def test_unified_prediction_interface(self, sample_features):
        """統一された予測インターフェースをテスト"""
        # 単一モデル（EnsembleTrainer with models=["lightgbm"]）
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer"):
            service_single = MLTrainingService(
                trainer_type="single", single_model_config={"model_type": "lightgbm"}
            )
            service_single.trainer = MagicMock(spec=EnsembleTrainer)
            service_single.trainer.is_trained = True
            service_single.trainer.is_single_model = True
            service_single.trainer.predict.return_value = np.random.rand(
                len(sample_features), 3
            )

            result_single = service_single.predict(sample_features)
            assert "predictions" in result_single

        # アンサンブルモデル
        with patch("backend.app.services.ml.ml_training_service.EnsembleTrainer"):
            service_ensemble = MLTrainingService(
                trainer_type="ensemble",
                ensemble_config={"models": ["lightgbm", "xgboost"]},
            )
            service_ensemble.trainer = MagicMock(spec=EnsembleTrainer)
            service_ensemble.trainer.is_trained = True
            service_ensemble.trainer.is_single_model = False
            service_ensemble.trainer.predict.return_value = np.random.rand(
                len(sample_features), 3
            )

            result_ensemble = service_ensemble.predict(sample_features)
            assert "predictions" in result_ensemble

        # 両方とも同じインターフェース
        assert result_single.keys() == result_ensemble.keys()
