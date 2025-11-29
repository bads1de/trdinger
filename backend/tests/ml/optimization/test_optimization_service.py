import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.services.optimization.optimization_service import (
    OptimizationService,
    OptimizationSettings,
)
from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer


class TestOptimizationService:
    @pytest.fixture
    def service(self):
        return OptimizationService()

    @pytest.fixture
    def mock_trainer(self):
        """EnsembleTrainer（単一モデルモード）のモック"""
        trainer = MagicMock(spec=EnsembleTrainer)
        trainer.train_model.return_value = {"f1_score": 0.8}
        trainer.is_single_model = True
        trainer.ensemble_config = {"method": "stacking", "models": ["lightgbm"]}
        return trainer

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [100, 101, 102],
                "volume": [10, 20, 30],
            }
        )

    def test_optimize_parameters(self, mock_trainer, sample_data):
        settings = OptimizationSettings(enabled=True, n_calls=2)

        with patch(
            "app.services.optimization.optimization_service.OptunaOptimizer"
        ) as MockOptimizer:
            # Setup mock BEFORE service init
            mock_opt_instance = MockOptimizer.return_value
            mock_opt_instance.optimize.return_value = MagicMock(
                best_params={"learning_rate": 0.05},
                best_score=0.85,
                total_evaluations=2,
                optimization_time=1.0,
            )
            mock_opt_instance.get_default_parameter_space.return_value = {}

            # Initialize service inside patch context or inject mock
            service = OptimizationService()
            # Ensure the service uses the mock instance
            service.optimizer = mock_opt_instance

            result = service.optimize_parameters(
                trainer=mock_trainer,
                training_data=sample_data,
                optimization_settings=settings,
                model_name="test_model",
            )

            assert result["best_params"] == {"learning_rate": 0.05}
            assert result["best_score"] == 0.85
            mock_opt_instance.optimize.assert_called_once()

    def test_create_objective_function(self, service, mock_trainer, sample_data):
        settings = OptimizationSettings(enabled=True, n_calls=2)

        objective = service._create_objective_function(
            trainer=mock_trainer,
            training_data=sample_data,
            optimization_settings=settings,
        )

        # Test the objective function
        # EnsembleTrainerを使用する（統一後）
        with patch(
            "app.services.optimization.optimization_service.EnsembleTrainer"
        ) as MockTrainerClass:
            mock_temp_trainer = MockTrainerClass.return_value
            mock_temp_trainer.train_model.return_value = {"f1_score": 0.75}

            score = objective({"learning_rate": 0.1})

            assert score == 0.75
            mock_temp_trainer.train_model.assert_called_once()
