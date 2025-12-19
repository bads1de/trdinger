import pytest
import numpy as np
from app.services.ml.models.gru_model import GRUModel

class TestGRUModel:
    @pytest.fixture
    def sample_data(self):
        # 50件、2特徴量
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_fit_and_predict(self, sample_data):
        """GRUモデルの学習と予測"""
        X, y = sample_data
        model = GRUModel(input_dim=2, seq_len=5, epochs=2, batch_size=8)
        
        model.fit(X, y)
        assert model.model is not None
        
        probs = model.predict_proba(X)
        assert len(probs) == len(X)
        assert np.all((probs >= 0) & (probs <= 1))
        
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})
