import pytest
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
import numpy as np
from backend.app.services.ml.ml_training_service import (
    MLTrainingService,
    OptimizationSettings,
)
from backend.app.services.ml.single_model.single_model_trainer import SingleModelTrainer
from backend.app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer


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

    @patch("backend.app.services.ml.ml_training_service.SingleModelTrainer")
    def test_optimization_single_model(self, mock_single_trainer_cls, sample_data):
        """Test that optimization uses SingleModelTrainer when configured"""
        # Setup
        service = MLTrainingService(
            trainer_type="single", single_model_config={"model_type": "lightgbm"}
        )
        opt_settings = OptimizationSettings(enabled=True, n_calls=1)

        # Configure service.trainer (which is a mock) to pass duck typing checks
        # Ensure ensemble_config does not exist
        del service.trainer.ensemble_config
        # Ensure model_type exists
        service.trainer.model_type = "lightgbm"

        # Mock the trainer instance created inside objective function
        mock_temp_trainer = MagicMock()
        mock_temp_trainer.train_model.return_value = {"f1_score": 0.8}
        mock_single_trainer_cls.return_value = mock_temp_trainer

        # Create objective function
        objective_func = service._create_objective_function(
            training_data=sample_data,
            optimization_settings=opt_settings,
            trainer=service.trainer,  # Pass the current trainer
        )

        # Execute objective function
        score = objective_func({"learning_rate": 0.05})

        # Verify
        assert score == 0.8
        # Verify SingleModelTrainer was instantiated with correct type
        mock_single_trainer_cls.assert_called_with(model_type="lightgbm")
        # Verify train_model was called with optimization params
        mock_temp_trainer.train_model.assert_called()
        call_kwargs = mock_temp_trainer.train_model.call_args[1]
        assert call_kwargs["learning_rate"] == 0.05
        assert call_kwargs["save_model"] is False

    @patch("backend.app.services.ml.ml_training_service.EnsembleTrainer")
    def test_optimization_ensemble_model(self, mock_ensemble_trainer_cls, sample_data):
        """Test that optimization uses EnsembleTrainer when configured"""
        # Setup
        ensemble_config = {
            "method": "stacking",
            "stacking_params": {"base_models": ["lightgbm"], "cv_folds": 5},
        }
        service = MLTrainingService(
            trainer_type="ensemble", ensemble_config=ensemble_config
        )
        opt_settings = OptimizationSettings(enabled=True, n_calls=1)

        # Configure service.trainer to pass duck typing checks
        # Ensure ensemble_config exists and is a dict (not a mock)
        service.trainer.ensemble_config = ensemble_config

        # Mock the trainer instance
        mock_temp_trainer = MagicMock()
        mock_temp_trainer.train_model.return_value = {"f1_score": 0.85}
        mock_ensemble_trainer_cls.return_value = mock_temp_trainer

        # Create objective function
        objective_func = service._create_objective_function(
            training_data=sample_data,
            optimization_settings=opt_settings,
            trainer=service.trainer,
        )

        # Execute objective function
        score = objective_func({"meta_model_learning_rate": 0.05})

        # Verify
        assert score == 0.85
        # Verify EnsembleTrainer was instantiated
        # Note: call_count might be > 1 because MLTrainingService init also creates one

        # Check the last call (which should be inside objective function)
        # We need to find the call that has the modified cv_folds
        found_call = False
        for call in mock_ensemble_trainer_cls.call_args_list:
            _, kwargs = call
            if "ensemble_config" in kwargs:
                config = kwargs["ensemble_config"]
                if config.get("stacking_params", {}).get("cv_folds") == 3:
                    found_call = True
                    break

        assert (
            found_call
        ), "EnsembleTrainer should be initialized with cv_folds=3 during optimization"

        # Verify train_model was called
        mock_temp_trainer.train_model.assert_called()
