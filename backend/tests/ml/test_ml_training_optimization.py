import pytest
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
import numpy as np
from backend.app.services.ml.ml_training_service import MLTrainingService
from backend.app.services.ml.optimization.optimization_service import (
    OptimizationSettings,
)


class TestMLTrainingOptimization:
    """ML学習最適化のテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """サンプルデータを生成"""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": np.random.rand(10),
                "high": np.random.rand(10),
                "low": np.random.rand(10),
                "close": np.random.rand(10),
                "volume": np.random.rand(10),
                "target": np.random.randint(0, 2, 10),
            },
            index=dates,
        )
        return df

    def test_train_model_with_optimization(self, sample_data):
        """最適化が有効な場合、train_modelがOptimizationServiceに処理を委譲することを確認"""
        # セットアップ
        service = MLTrainingService(
            trainer_type="single", single_model_config={"model_type": "lightgbm"}
        )
        opt_settings = OptimizationSettings(enabled=True, n_calls=1)

        # OptimizationServiceをモック化
        service.optimization_service = MagicMock()
        service.optimization_service.optimize_parameters.return_value = {
            "best_params": {"learning_rate": 0.05},
            "best_score": 0.85,
            "total_evaluations": 1,
            "optimization_time": 1.0,
        }

        # トレーナーをモック化
        service.trainer = MagicMock()
        service.trainer.train_model.return_value = {"f1_score": 0.9}

        # 実行
        result = service.train_model(
            training_data=sample_data,
            optimization_settings=opt_settings,
            save_model=False,
        )

        # OptimizationServiceが呼び出されたことを確認
        service.optimization_service.optimize_parameters.assert_called_once()
        call_kwargs = service.optimization_service.optimize_parameters.call_args[1]
        assert call_kwargs["trainer"] == service.trainer
        assert call_kwargs["training_data"] is sample_data
        assert call_kwargs["optimization_settings"] == opt_settings

        # 最適なパラメータで最終学習が呼び出されたことを確認
        service.trainer.train_model.assert_called_once()
        train_kwargs = service.trainer.train_model.call_args[1]
        assert train_kwargs["learning_rate"] == 0.05
        assert train_kwargs["save_model"] is False

        # 結果に最適化情報が含まれていることを確認
        assert "optimization_result" in result
        assert result["optimization_result"]["best_params"] == {"learning_rate": 0.05}

    def test_train_model_without_optimization(self, sample_data):
        """最適化が無効な場合、train_modelがOptimizationServiceをバイパスすることを確認"""
        # セットアップ
        service = MLTrainingService(
            trainer_type="single", single_model_config={"model_type": "lightgbm"}
        )
        opt_settings = OptimizationSettings(enabled=False)

        # OptimizationServiceをモック化
        service.optimization_service = MagicMock()

        # トレーナーをモック化
        service.trainer = MagicMock()
        service.trainer.train_model.return_value = {"f1_score": 0.9}

        # 実行
        service.train_model(
            training_data=sample_data,
            optimization_settings=opt_settings,
            save_model=False,
        )

        # OptimizationServiceが呼び出されていないことを確認
        service.optimization_service.optimize_parameters.assert_not_called()

        # トレーニングが呼び出されたことを確認
        service.trainer.train_model.assert_called_once()



