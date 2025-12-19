import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.orchestration.ml_training_service import MLTrainingService
from app.services.ml.optimization.optimization_service import OptimizationSettings

class TestMLTrainingService:
    @pytest.fixture
    def service(self):
        # 実際には EnsembleTrainer が初期化されるが、テストではモック化する
        with patch('app.services.ml.orchestration.ml_training_service.EnsembleTrainer'):
            return MLTrainingService(trainer_type="single", single_model_config={"model_type": "lightgbm"})

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame({"close": np.random.randn(100) + 100}, index=dates)
        return df

    def test_init_single(self):
        """単一モデル設定での初期化テスト"""
        with patch('app.services.ml.orchestration.ml_training_service.EnsembleTrainer') as mock_trainer:
            service = MLTrainingService(trainer_type="single", single_model_config={"model_type": "xgboost"})
            assert service.trainer_type == "single"
            # 内部で EnsembleTrainer が適切な設定で呼ばれたか
            args, kwargs = mock_trainer.call_args
            assert kwargs["ensemble_config"]["model_type"] == "xgboost"

    def test_init_ensemble_default(self):
        """アンサンブル設定での初期化テスト（デフォルト）"""
        with patch('app.services.ml.orchestration.ml_training_service.EnsembleTrainer'):
            service = MLTrainingService(trainer_type="ensemble")
            assert service.trainer_type == "ensemble"

    def test_train_model_success(self, service, sample_data):
        """正常な学習フローのテスト"""
        service.trainer.train_model.return_value = {"accuracy": 0.8}
        
        result = service.train_model(sample_data, save_model=False)
        
        assert result["accuracy"] == 0.8
        assert result["is_optimized"] is False
        service.trainer.train_model.assert_called_once()

    def test_train_model_with_optimization(self, service, sample_data):
        """最適化ありの学習フロー"""
        settings = OptimizationSettings(enabled=True, n_calls=2)
        service.optimization_service.optimize_parameters = MagicMock(return_value={
            "best_params": {"n_estimators": 100},
            "best_score": 0.85
        })
        service.trainer.train_model.return_value = {"accuracy": 0.85}
        
        result = service.train_model(sample_data, optimization_settings=settings)
        
        assert result["is_optimized"] is True
        assert "optimization_result" in result
        # 最適化されたパラメータが渡されたか
        args, kwargs = service.trainer.train_model.call_args
        assert kwargs["n_estimators"] == 100

    def test_predict_and_signals(self, service, sample_data):
        """予測とシグナル生成のテスト"""
        service.trainer.predict.return_value = np.array([0.1, 0.9])
        service.trainer.predict_signal.return_value = {"is_valid": 0.9}
        
        res_pred = service.predict(sample_data)
        assert "predictions" in res_pred
        
        res_sig = service.generate_signals(sample_data)
        assert res_sig["is_valid"] == 0.9

    def test_load_model(self, service):
        """モデル読み込みの委譲"""
        service.trainer.load_model.return_value = True
        assert service.load_model("/path") is True
        service.trainer.load_model.assert_called_with("/path")

    def test_get_available_single_models(self):
        """利用可能なモデルリストの取得"""
        models = MLTrainingService.get_available_single_models()
        assert "lightgbm" in models
        assert "xgboost" in models
