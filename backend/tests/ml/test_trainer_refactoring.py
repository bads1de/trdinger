import pytest
from unittest.mock import MagicMock, patch
from backend.app.services.ml.single_model.single_model_trainer import SingleModelTrainer
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
        """Test SingleModelTrainer.save_model after refactoring"""
        trainer = SingleModelTrainer(model_type="lightgbm")
        trainer.is_trained = True
        trainer.feature_columns = ["f1", "f2"]
        trainer.single_model = MagicMock()
        trainer.single_model.is_trained = True
        trainer.single_model.model = MagicMock()
        # Set _model as it is done in _train_model_impl
        trainer._model = trainer.single_model.model

        # Ensure feature_importance attribute does not exist
        del trainer._model.feature_importance

        # Mock get_feature_importance
        # BaseMLTrainer.get_feature_importance uses self._model.feature_importance
        # trainer._model.feature_importance.return_value = [0.5, 0.5]
        # Or if it uses get_feature_importance method
        trainer._model.get_feature_importance.return_value = {
            "f1": 0.5,
            "f2": 0.5,
        }

        mock_single_model_save.save_model.return_value = "path/to/model"

        path = trainer.save_model("test_model")

        assert path == "path/to/model"
        args, kwargs = mock_single_model_save.save_model.call_args
        metadata = kwargs["metadata"]
        # BaseMLTrainer sets model_type to __class__.__name__, but SingleModelTrainer overrides it
        assert metadata["model_type"] == "lightgbm"
        assert metadata["is_trained"] is True
        assert "feature_count" in metadata
        # feature_importance might be in metadata if get_feature_importance works
        # BaseMLTrainer calls get_feature_importance(top_n=100)

    def test_ensemble_model_save_model_duplication(self, mock_ensemble_model_save):
        """Test EnsembleTrainer.save_model after refactoring"""
        config = {"method": "stacking"}
        trainer = EnsembleTrainer(ensemble_config=config)
        trainer.is_trained = True
        trainer.feature_columns = ["f1", "f2"]
        trainer.ensemble_model = MagicMock()
        trainer.ensemble_model.is_fitted = True
        trainer.ensemble_model.best_algorithm = "lgbm"
        # Set _model
        trainer._model = trainer.ensemble_model

        # Ensure feature_importance attribute does not exist so it falls through to get_feature_importance method
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
        # Single Model
        trainer = SingleModelTrainer(model_type="lightgbm")
        trainer.is_trained = True
        trainer.single_model = MagicMock()
        trainer._model = MagicMock()  # Set _model

        # Ensure feature_importance attribute does not exist
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
        trainer_ens = EnsembleTrainer(ensemble_config={})
        trainer_ens.is_trained = True
        trainer_ens.ensemble_model = MagicMock()
        trainer_ens._model = trainer_ens.ensemble_model  # Set _model

        # Ensure feature_importance attribute does not exist
        del trainer_ens.ensemble_model.feature_importance

        trainer_ens.ensemble_model.get_feature_importance.return_value = {
            "f1": 0.7,
            "f2": 0.3,
        }
        trainer_ens.feature_columns = ["f1", "f2"]

        fi_ens = trainer_ens.get_feature_importance()
        assert fi_ens == {"f1": 0.7, "f2": 0.3}
