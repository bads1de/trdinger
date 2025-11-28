import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from backend.app.services.ml.ml_training_service import MLTrainingService

class TestRefactorReproduction:
    """リファクタリングの再現テスト"""

    @pytest.fixture
    def mock_config(self):
        with patch("app.services.ml.config.ml_config") as mock_config:
            # training属性をMagicMockで上書き
            mock_config.training = MagicMock()
            
            # Mock the structure: config.training.label_generation
            mock_config.training.USE_PURGED_KFOLD = False
            mock_config.training.CROSS_VALIDATION_FOLDS = 5
            mock_config.training.MAX_TRAIN_SIZE = None
            mock_config.training.USE_TIME_SERIES_SPLIT = True
            mock_config.training.PREDICTION_HORIZON = 24
            
            # Mock label generation config
            mock_label_gen = MagicMock()
            mock_label_gen.use_preset = False # デフォルトの挙動を制御
            mock_config.training.label_generation = mock_label_gen
            
            yield mock_config

    @pytest.fixture
    def sample_training_data(self):
        """サンプル学習データ"""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=2113, freq="h")
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.randn(2113) * 100 + 10000,
            "high": np.random.randn(2113) * 100 + 10100,
            "low": np.random.randn(2113) * 100 + 9900,
            "close": np.random.randn(2113) * 100 + 10000,
            "volume": np.random.randint(100, 1000, 2113),
        })
        return df.set_index("timestamp")

    def test_single_model_training(self, sample_training_data, mock_config):
        """SingleModelTrainerがMLTrainingService経由で動作することを確認"""
        service = MLTrainingService(
            trainer_type="single",
            single_model_config={"model_type": "lightgbm"}
        )

        # テスト用にパラメータを緩和
        result = service.train_model(
            sample_training_data,
            save_model=False,
            # 必須カラム以外のパラメータ
            target_column="close", # ダミー
            use_cross_validation=False,
            use_time_series_split=False,
            test_size=0.2
        )

        assert result["success"] is True
        assert result["model_type"] == "lightgbm"

    def test_ensemble_model_training(self, sample_training_data, mock_config):
        """EnsembleTrainerがMLTrainingService経由で動作することを確認"""
        service = MLTrainingService(
            trainer_type="ensemble",
            ensemble_config={
                "method": "stacking",
                "stacking_params": {
                    "base_models": ["lightgbm"], # 高速化のため1つ
                    "meta_model": "lightgbm",
                    "cv_folds": 2,
                    "cv_strategy": "stratified_kfold" # テスト用に層化KFoldを指定
                }
            }
        )

        result = service.train_model(
            sample_training_data,
            save_model=False,
            target_column="close", # ダミー
            use_cross_validation=False
        )

        assert result["success"] is True
        assert result["model_type"] == "EnsembleModel"
