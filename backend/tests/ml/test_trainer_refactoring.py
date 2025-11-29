import pytest
from unittest.mock import MagicMock, patch
from backend.app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer


class TestTrainerRefactoring:

    @pytest.fixture
    def mock_model_manager(self):
        # Patch where it is used in BaseMLTrainer
        with patch("backend.app.services.ml.base_ml_trainer.model_manager") as mock:
            yield mock

    @pytest.fixture
    def mock_single_model_save(self, mock_model_manager):
        return mock_model_manager

    @pytest.fixture
    def mock_ensemble_model_save(self, mock_model_manager):
        return mock_model_manager

    def test_single_model_save_model_duplication(self, mock_single_model_save):
        """Test EnsembleTrainer (single model mode) save_model after refactoring"""
        # 単一モデルモードでEnsembleTrainerを使用
        config = {"method": "stacking", "models": ["lightgbm"]}
        trainer = EnsembleTrainer(ensemble_config=config)
        trainer.is_trained = True
        trainer.feature_columns = ["f1", "f2"]
        trainer.ensemble_model = MagicMock()
        trainer.ensemble_model.is_fitted = True
        trainer.ensemble_model.model = MagicMock()
        trainer._model = trainer.ensemble_model

        # Ensure feature_importance attribute does not exist
        if hasattr(trainer._model, "feature_importance"):
            del trainer._model.feature_importance

        # Mock get_feature_importance
        trainer._model.get_feature_importance.return_value = {
            "f1": 0.5,
            "f2": 0.5,
        }

        mock_single_model_save.save_model.return_value = "path/to/model"

        path = trainer.save_model("test_model")

        assert path == "path/to/model"
        args, kwargs = mock_single_model_save.save_model.call_args
        metadata = kwargs["metadata"]
        # EnsembleTrainer（単一モデルモード）はmodel_typeを設定
        assert "model_type" in metadata
        assert metadata["is_trained"] is True
        assert "feature_count" in metadata

    def test_ensemble_model_save_model_duplication(self, mock_ensemble_model_save):
        """Test EnsembleTrainer.save_model after refactoring"""
        config = {"method": "stacking", "models": ["lightgbm", "xgboost"]}
        trainer = EnsembleTrainer(ensemble_config=config)
        trainer.is_trained = True
        trainer.feature_columns = ["f1", "f2"]
        trainer.ensemble_model = MagicMock()
        trainer.ensemble_model.is_fitted = True
        trainer.ensemble_model.best_algorithm = "lgbm"
        # Set _model
        trainer._model = trainer.ensemble_model

        # Ensure feature_importance attribute does not exist so it falls through to get_feature_importance method
        if hasattr(trainer.ensemble_model, "feature_importance"):
            del trainer.ensemble_model.feature_importance

        # Mock get_feature_importance
        trainer.ensemble_model.get_feature_importance.return_value = {
            "f1": 0.6,
            "f2": 0.4,
        }

        mock_ensemble_model_save.save_model.return_value = "path/to/ensemble_pickle"

        path = trainer.save_model("test_ensemble")

        assert path == "path/to/ensemble_pickle"

        # Verify save_model was called with correct metadata
        args, kwargs = mock_ensemble_model_save.save_model.call_args
        metadata = kwargs["metadata"]
        # BaseMLTrainer sets model_type to __class__.__name__, but EnsembleTrainer overrides it via metadata
        assert metadata["model_type"] == "lgbm"
        assert metadata["is_trained"] is True
        assert "feature_count" in metadata
        assert "feature_importance" in metadata

        # Verify model object passed is the trainer itself (BaseMLTrainer saves 'self')
        assert kwargs["model"] == trainer.ensemble_model

    def test_get_feature_importance_duplication(self):
        """Test get_feature_importance duplication"""
        # Single Model mode (using EnsembleTrainer)
        config_single = {"method": "stacking", "models": ["lightgbm"]}
        trainer = EnsembleTrainer(ensemble_config=config_single)
        trainer.is_trained = True
        trainer.ensemble_model = MagicMock()
        trainer._model = trainer.ensemble_model  # Set _model

        # Ensure feature_importance attribute does not exist
        if hasattr(trainer._model, "feature_importance"):
            del trainer._model.feature_importance

        # Mock for BaseMLTrainer.get_feature_importance logic
        # It checks hasattr(self._model, "feature_importance") or "get_feature_importance"
        trainer._model.get_feature_importance.return_value = {
            "f1": 0.8,
            "f2": 0.2,
        }
        trainer.feature_columns = ["f1", "f2"]

        fi = trainer.get_feature_importance()
        assert fi == {"f1": 0.8, "f2": 0.2}

        # Ensemble Model
        config_ensemble = {"method": "stacking", "models": ["lightgbm", "xgboost"]}
        trainer_ens = EnsembleTrainer(ensemble_config=config_ensemble)
        trainer_ens.is_trained = True
        trainer_ens.ensemble_model = MagicMock()
        trainer_ens._model = trainer_ens.ensemble_model  # Set _model

        # Ensure feature_importance attribute does not exist
        if hasattr(trainer_ens.ensemble_model, "feature_importance"):
            del trainer_ens.ensemble_model.feature_importance

        trainer_ens.ensemble_model.get_feature_importance.return_value = {
            "f1": 0.7,
            "f2": 0.3,
        }
        trainer_ens.feature_columns = ["f1", "f2"]

        fi_ens = trainer_ens.get_feature_importance()
        assert fi_ens == {"f1": 0.7, "f2": 0.3}
