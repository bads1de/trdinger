import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from backend.app.services.ml.models.lightgbm import LightGBMModel
from backend.app.services.ml.models.xgboost import XGBoostModel
from backend.app.services.ml.models.catboost import CatBoostModel


class TestGradientBoostingModels:

    @pytest.fixture
    def sample_data(self):
        X = pd.DataFrame({"f1": np.random.rand(10), "f2": np.random.rand(10)})
        y = pd.Series(np.random.randint(0, 2, 10))
        return X, y

    def test_lightgbm_predict_logic(self, sample_data):
        """Test LightGBMModel predict logic via BaseGradientBoostingModel"""
        X, y = sample_data
        model = LightGBMModel()
        model.is_trained = True
        model.feature_columns = ["f1", "f2"]
        model.model = MagicMock()
        model.model.best_iteration = 100

        # Mock predict output (probabilities)
        # Case 1: Binary classification (1D array from lightgbm)
        mock_probas = np.array([0.1, 0.8, 0.4, 0.6, 0.2, 0.9, 0.3, 0.7, 0.1, 0.9])
        model.model.predict.return_value = mock_probas

        # Test predict_proba
        probas = model.predict_proba(X)
        assert probas.shape == (10, 2)
        assert np.allclose(probas[:, 1], mock_probas)
        assert np.allclose(probas[:, 0], 1 - mock_probas)

        # Test predict
        preds = model.predict(X)
        assert preds.shape == (10,)
        expected_preds = (mock_probas > 0.5).astype(int)
        assert np.array_equal(preds, expected_preds)

        # Verify _prepare_input_for_prediction was called (implicitly, by checking args passed to mock)
        # LightGBMModel passes X directly
        model.model.predict.assert_called_with(X, num_iteration=100)

    def test_xgboost_predict_logic(self, sample_data):
        """Test XGBoostModel predict logic via BaseGradientBoostingModel"""
        X, y = sample_data
        model = XGBoostModel()
        model.is_trained = True
        model.feature_columns = ["f1", "f2"]
        model.feature_names = ["f1", "f2"]
        model.model = MagicMock()

        # Mock predict output
        # Case 1: Binary classification (1D array from xgboost)
        mock_probas = np.array([0.2, 0.7, 0.3, 0.8, 0.1, 0.95, 0.4, 0.6, 0.15, 0.85])
        model.model.predict.return_value = mock_probas

        # Test predict_proba
        probas = model.predict_proba(X)
        assert probas.shape == (10, 2)
        assert np.allclose(probas[:, 1], mock_probas)

        # Test predict
        preds = model.predict(X)
        assert preds.shape == (10,)
        expected_preds = (mock_probas > 0.5).astype(int)
        assert np.array_equal(preds, expected_preds)

        # Verify _prepare_input_for_prediction used DMatrix
        # We can't easily check the DMatrix content without mocking xgb.DMatrix,
        # but we can check that model.predict was called with something
        assert model.model.predict.called

    def test_multiclass_prediction(self, sample_data):
        """Test multiclass prediction logic"""
        X, _ = sample_data
        model = LightGBMModel()
        model.is_trained = True
        model.feature_columns = ["f1", "f2"]
        model.model = MagicMock()
        model.model.best_iteration = 100

        # Mock 3-class probabilities
        mock_probas = np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] * 3 + [[0.3, 0.3, 0.4]]
        )  # 10 samples
        model.model.predict.return_value = mock_probas

        # Test predict_proba
        probas = model.predict_proba(X)
        assert probas.shape == (10, 3)
        assert np.array_equal(probas, mock_probas)

        # Test predict
        preds = model.predict(X)
        assert preds.shape == (10,)
        expected_preds = np.argmax(mock_probas, axis=1)
        assert np.array_equal(preds, expected_preds)

    def test_catboost_predict_logic(self, sample_data):
        """Test CatBoostModel predict logic (now inherits from BaseGradientBoostingModel)"""
        X, y = sample_data
        model = CatBoostModel()
        model.is_trained = True  # BaseGradientBoostingModelの規約に合わせる
        model.feature_columns = ["f1", "f2"]
        model.model = MagicMock()

        # Mock predict_proba output
        mock_probas = np.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.3, 0.7],
                [0.9, 0.1],
                [0.4, 0.6],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.75, 0.25],
                [0.35, 0.65],
            ]
        )
        model.model.predict_proba.return_value = mock_probas

        # Test predict_proba
        probas = model.predict_proba(X)
        assert probas.shape == (10, 2)
        assert np.array_equal(probas, mock_probas)

        # Test predict (derived from predict_proba)
        preds = model.predict(X)
        assert preds.shape == (10,)
        expected_preds = np.argmax(mock_probas, axis=1)
        assert np.array_equal(preds, expected_preds)
