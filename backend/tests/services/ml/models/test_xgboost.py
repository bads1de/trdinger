import pytest
import pandas as pd
import numpy as np
from app.services.ml.models.xgboost import XGBoostModel

class TestXGBoostModel:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 50
        X = pd.DataFrame(np.random.randn(n, 2), columns=["f1", "f2"])
        y = (X["f1"] + X["f2"] > 0).astype(int)
        return X, y

    def test_fit_and_predict(self, sample_data):
        """XGBoostの学習と予測（モック化）"""
        X, y = sample_data
        model = XGBoostModel(n_estimators=10)
        
        from unittest.mock import MagicMock, patch
        with patch('app.services.ml.models.xgboost.xgb.train') as mock_train:
            with patch('app.services.ml.models.xgboost.xgb.DMatrix') as mock_dmatrix:
                mock_booster = mock_train.return_value
                mock_booster.predict.return_value = np.array([0.8])
                
                model.fit(X, y)
                assert model.is_trained is True
                
                probs = model.predict_proba(X.iloc[:1])
                assert probs[0, 1] == 0.8

    def test_early_stopping(self, sample_data):
        """XGBoost.Early Stoppingのテスト（モック化）"""
        X, y = sample_data
        model = XGBoostModel(n_estimators=50)
        
        from unittest.mock import MagicMock, patch
        with patch('app.services.ml.models.xgboost.xgb.train') as mock_train:
            with patch('app.services.ml.models.xgboost.xgb.DMatrix'):
                mock_booster = mock_train.return_value
                mock_booster.best_iteration = 10
                
                # 内部評価用の予測確率をモック化
                with patch.object(model, '_get_prediction_proba', return_value=np.zeros(len(y)//5)):
                    model.fit(X, y, early_stopping_rounds=5)
                    assert model.is_trained is True
                    assert mock_train.called
                    args, kwargs = mock_train.call_args
                    assert kwargs["early_stopping_rounds"] == 5

    def test_multiclass(self):
        """多クラス分類（モック化）"""
        X = pd.DataFrame(np.random.randn(60, 2), columns=["f1", "f2"])
        y = np.random.randint(0, 3, 60)
        model = XGBoostModel(n_estimators=5)
        
        from unittest.mock import MagicMock, patch
        with patch('app.services.ml.models.xgboost.xgb.train'):
            with patch('app.services.ml.models.xgboost.xgb.DMatrix'):
                model.fit(X, y)
                assert model.is_trained is True
