from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.services.ml.models.xgboost import XGBoostModel


class TestXGBoostModel:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 50
        X = pd.DataFrame(np.random.randn(n, 2), columns=["f1", "f2"])
        # 回帰用の連続ターゲット
        y = X["f1"] + X["f2"] + np.random.randn(n) * 0.1
        return X, y

    def test_fit_and_predict(self, sample_data):
        """XGBoostの学習と予測（モック化）"""
        X, y = sample_data
        model = XGBoostModel(n_estimators=10)

        with patch("app.services.ml.models.xgboost.xgb.train") as mock_train:
            with patch("app.services.ml.models.xgboost.xgb.DMatrix"):
                mock_booster = mock_train.return_value
                # 回帰なので1次元の連続値
                mock_booster.predict.return_value = np.array([0.5])

                model.fit(X, y)
                assert model.is_trained is True

                # 回帰タスクでは predict_proba は (n, 1) を返す
                preds = model.predict_proba(X.iloc[:1])
                assert preds.shape == (1, 1)
                assert preds[0, 0] == 0.5

    def test_early_stopping(self, sample_data):
        """XGBoost.Early Stoppingのテスト（モック化）"""
        X, y = sample_data
        model = XGBoostModel(n_estimators=50)

        with patch("app.services.ml.models.xgboost.xgb.train") as mock_train:
            with patch("app.services.ml.models.xgboost.xgb.DMatrix"):
                mock_booster = mock_train.return_value
                mock_booster.best_iteration = 10
                # 回帰なので1次元の連続値を返す
                mock_booster.predict.return_value = np.array([0.5] * (len(y) // 5))

                model.fit(X, y, early_stopping_rounds=5)
                assert model.is_trained is True
                assert mock_train.called
                args, kwargs = mock_train.call_args
                assert kwargs["early_stopping_rounds"] == 5
