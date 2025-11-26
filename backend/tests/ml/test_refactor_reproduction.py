import pytest
import pandas as pd
import numpy as np
from backend.app.services.ml.ml_training_service import MLTrainingService

class TestRefactorReproduction:
    @pytest.fixture
    def sample_training_data(self):
        """実際のトレーニングデータに近いダミーデータ（データ量を増やして安定化）"""
        np.random.seed(42)
        # データ量を増やす (約2000行)
        dates = pd.date_range(start="2023-01-01", end="2023-03-30", freq="1h")

        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": 10000 + np.cumsum(np.random.randn(len(dates)) * 10), # ランダムウォークで少し現実に近づける
                "volume": 500 + np.random.randint(100, 1000, len(dates)),
            }
        )
        
        # OHLCの関係を確保
        data["close"] = data["open"] + np.random.randn(len(dates)) * 10
        data["high"] = data[["open", "close"]].max(axis=1) + np.abs(np.random.randn(len(dates)) * 5)
        data["low"] = data[["open", "close"]].min(axis=1) - np.abs(np.random.randn(len(dates)) * 5)

        return data

    def test_single_model_training(self, sample_training_data):
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
        assert "accuracy" in result or "f1_score" in result or "rmse" in result # 指標はタスクによる

    def test_ensemble_model_training(self, sample_training_data):
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
        assert "ensemble" in str(result.get("model_type", "")).lower() or "stacking" in str(result.get("ensemble_method", "")).lower()
