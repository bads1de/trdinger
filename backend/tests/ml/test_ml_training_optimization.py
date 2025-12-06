import pytest
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
import numpy as np
from backend.app.services.ml.ml_training_service import MLTrainingService
from backend.app.services.ml.optimization.optimization_service import (
    OptimizationSettings,
)


class TestMLTrainingOptimization:

    @pytest.fixture
    def sample_data(self):
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
        """Test that train_model delegates to OptimizationService when optimization is enabled"""
        # Setup
        service = MLTrainingService(
            trainer_type="single", single_model_config={"model_type": "lightgbm"}
        )
        opt_settings = OptimizationSettings(enabled=True, n_calls=1)

        # Mock OptimizationService
        service.optimization_service = MagicMock()
        service.optimization_service.optimize_parameters.return_value = {
            "best_params": {"learning_rate": 0.05},
            "best_score": 0.85,
            "total_evaluations": 1,
            "optimization_time": 1.0,
        }

        # Mock trainer
        service.trainer = MagicMock()
        service.trainer.train_model.return_value = {"f1_score": 0.9}

        # Execute
        result = service.train_model(
            training_data=sample_data,
            optimization_settings=opt_settings,
            save_model=False,
        )

        # Verify OptimizationService called
        service.optimization_service.optimize_parameters.assert_called_once()
        call_kwargs = service.optimization_service.optimize_parameters.call_args[1]
        assert call_kwargs["trainer"] == service.trainer
        assert call_kwargs["training_data"] is sample_data
        assert call_kwargs["optimization_settings"] == opt_settings

        # Verify final training called with best params
        service.trainer.train_model.assert_called_once()
        train_kwargs = service.trainer.train_model.call_args[1]
        assert train_kwargs["learning_rate"] == 0.05
        assert train_kwargs["save_model"] is False

        # Verify result contains optimization info
        assert "optimization_result" in result
        assert result["optimization_result"]["best_params"] == {"learning_rate": 0.05}

    def test_train_model_without_optimization(self, sample_data):
        """Test that train_model bypasses OptimizationService when optimization is disabled"""
        # Setup
        service = MLTrainingService(
            trainer_type="single", single_model_config={"model_type": "lightgbm"}
        )
        opt_settings = OptimizationSettings(enabled=False)

        # Mock OptimizationService
        service.optimization_service = MagicMock()

        # Mock trainer
        service.trainer = MagicMock()
        service.trainer.train_model.return_value = {"f1_score": 0.9}

        # Execute
        service.train_model(
            training_data=sample_data,
            optimization_settings=opt_settings,
            save_model=False,
        )

        # Verify OptimizationService NOT called
        service.optimization_service.optimize_parameters.assert_not_called()

        # Verify training called
        service.trainer.train_model.assert_called_once()
