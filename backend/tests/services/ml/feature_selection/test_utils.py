"""
feature_selection.utils のテスト

LightGBM が利用可能な場合と、利用できない場合の両方で
デフォルト estimator の選択ロジックを確認する。
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.services.ml.feature_selection import utils as feature_selection_utils


class _DummyEstimator(BaseEstimator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        self.fit_shape_ = (len(X), len(X[0]) if len(X) else 0)
        return self


def test_get_default_estimator_prefers_lightgbm_when_available(monkeypatch):
    monkeypatch.setattr(feature_selection_utils, "LIGHTGBM_AVAILABLE", True)
    monkeypatch.setattr(feature_selection_utils, "LGBMClassifier", _DummyEstimator)

    model = feature_selection_utils.get_default_estimator(
        n_estimators=12,
        random_state=7,
        n_jobs=3,
    )

    assert isinstance(model, _DummyEstimator)
    assert isinstance(model, BaseEstimator)
    assert model.kwargs["n_estimators"] == 12
    assert model.kwargs["random_state"] == 7
    assert model.kwargs["n_jobs"] == 3


def test_get_default_regressor_prefers_lightgbm_when_available(monkeypatch):
    monkeypatch.setattr(feature_selection_utils, "LIGHTGBM_AVAILABLE", True)
    monkeypatch.setattr(feature_selection_utils, "LGBMRegressor", _DummyEstimator)

    model = feature_selection_utils.get_default_regressor(
        n_estimators=8,
        random_state=11,
        n_jobs=2,
    )

    assert isinstance(model, _DummyEstimator)
    assert isinstance(model, BaseEstimator)
    assert model.kwargs["n_estimators"] == 8
    assert model.kwargs["random_state"] == 11
    assert model.kwargs["n_jobs"] == 2


def test_get_task_appropriate_estimator_uses_classifier_for_classification_target(
    monkeypatch,
):
    monkeypatch.setattr(feature_selection_utils, "LIGHTGBM_AVAILABLE", False)
    monkeypatch.setattr(feature_selection_utils, "LGBMClassifier", None)
    monkeypatch.setattr(feature_selection_utils, "LGBMRegressor", None)

    y = np.array([0, 1, 0, 1, 0])
    model = feature_selection_utils.get_task_appropriate_estimator(y)

    assert isinstance(model, RandomForestClassifier)


def test_get_task_appropriate_estimator_uses_regressor_for_regression_target(
    monkeypatch,
):
    monkeypatch.setattr(feature_selection_utils, "LIGHTGBM_AVAILABLE", False)
    monkeypatch.setattr(feature_selection_utils, "LGBMClassifier", None)
    monkeypatch.setattr(feature_selection_utils, "LGBMRegressor", None)

    y = np.array([0.1, 1.2, 2.3, 3.4])
    model = feature_selection_utils.get_task_appropriate_estimator(y)

    assert isinstance(model, RandomForestRegressor)
