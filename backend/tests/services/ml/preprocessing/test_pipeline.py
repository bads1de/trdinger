"""
ML pipeline.py のユニットテスト

create_ml_pipeline / create_classification_pipeline / create_regression_pipeline
get_ml_pipeline_info / optimize_ml_pipeline をテストします。
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.services.ml.preprocessing.pipeline import (
    create_classification_pipeline,
    create_ml_pipeline,
    create_regression_pipeline,
    get_ml_pipeline_info,
    optimize_ml_pipeline,
)


@pytest.fixture
def sample_xy() -> tuple:
    """テスト用の X, y データ"""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(
        {f"f{i}": np.random.randn(n) for i in range(8)},
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    y = pd.Series(np.random.randint(0, 2, n), index=X.index)
    return X, y


# ---------------------------------------------------------------------------
# create_ml_pipeline
# ---------------------------------------------------------------------------

class TestCreateMlPipeline:
    def test_basic_pipeline_has_preprocessing(self):
        pipe = create_ml_pipeline()
        assert isinstance(pipe, Pipeline)
        names = [s[0] for s in pipe.steps]
        assert "preprocessing" in names

    def test_scaling_default_standard(self):
        pipe = create_ml_pipeline(scaling=True, scaling_method="standard")
        names = [s[0] for s in pipe.steps]
        assert "scaler" in names

    def test_scaling_robust(self):
        pipe = create_ml_pipeline(scaling=True, scaling_method="robust")
        names = [s[0] for s in pipe.steps]
        assert "scaler" in names

    def test_scaling_minmax(self):
        pipe = create_ml_pipeline(scaling=True, scaling_method="minmax")
        names = [s[0] for s in pipe.steps]
        assert "scaler" in names

    def test_no_scaling(self):
        pipe = create_ml_pipeline(scaling=False)
        names = [s[0] for s in pipe.steps]
        assert "scaler" not in names

    def test_invalid_scaling_raises(self):
        with pytest.raises(ValueError, match="スケーリング方法"):
            create_ml_pipeline(scaling=True, scaling_method="unknown")

    def test_feature_selection_f_regression(self):
        pipe = create_ml_pipeline(
            feature_selection=True, n_features=5, selection_method="f_regression"
        )
        names = [s[0] for s in pipe.steps]
        assert "feature_selection" in names

    def test_feature_selection_mutual_info(self):
        pipe = create_ml_pipeline(
            feature_selection=True, n_features=3, selection_method="mutual_info"
        )
        names = [s[0] for s in pipe.steps]
        assert "feature_selection" in names

    def test_invalid_selection_method_raises(self):
        with pytest.raises(ValueError, match="選択方法"):
            create_ml_pipeline(
                feature_selection=True, n_features=5, selection_method="invalid"
            )

    def test_no_feature_selection_when_n_features_none(self):
        pipe = create_ml_pipeline(feature_selection=True, n_features=None)
        names = [s[0] for s in pipe.steps]
        assert "feature_selection" not in names

    def test_no_feature_selection_when_n_features_zero(self):
        pipe = create_ml_pipeline(feature_selection=True, n_features=0)
        names = [s[0] for s in pipe.steps]
        assert "feature_selection" not in names

    def test_classification_uses_f_classif(self):
        pipe = create_ml_pipeline(
            is_classification=True,
            feature_selection=True,
            n_features=3,
            selection_method="f_classif",
        )
        names = [s[0] for s in pipe.steps]
        assert "feature_selection" in names

    def test_full_pipeline_steps(self):
        pipe = create_ml_pipeline(
            feature_selection=True,
            n_features=5,
            scaling=True,
            scaling_method="standard",
        )
        names = [s[0] for s in pipe.steps]
        assert "preprocessing" in names
        assert "feature_selection" in names
        assert "scaler" in names

    def test_fit_transform(self, sample_xy):
        """パイプラインが fit_transform できること"""
        X, y = sample_xy
        pipe = create_ml_pipeline(scaling=True, scaling_method="standard")
        result = pipe.fit_transform(X, y)
        assert result is not None
        assert len(result) == len(X)


# ---------------------------------------------------------------------------
# create_classification_pipeline / create_regression_pipeline
# ---------------------------------------------------------------------------

class TestConveniencePipelines:
    def test_classification_pipeline(self):
        pipe = create_classification_pipeline(feature_selection=True, n_features=3)
        assert isinstance(pipe, Pipeline)

    def test_regression_pipeline(self):
        pipe = create_regression_pipeline(feature_selection=True, n_features=3)
        assert isinstance(pipe, Pipeline)

    def test_classification_fit(self, sample_xy):
        X, y = sample_xy
        pipe = create_classification_pipeline(scaling=True)
        pipe.fit(X, y)
        assert pipe is not None


# ---------------------------------------------------------------------------
# get_ml_pipeline_info
# ---------------------------------------------------------------------------

class TestGetMlPipelineInfo:
    def test_returns_dict(self):
        pipe = create_ml_pipeline()
        info = get_ml_pipeline_info(pipe)
        assert isinstance(info, dict)
        assert info["pipeline_type"] == "ml"
        assert "n_steps" in info
        assert "step_names" in info

    def test_detects_preprocessing(self):
        pipe = create_ml_pipeline()
        info = get_ml_pipeline_info(pipe)
        assert info["has_preprocessing"] is True

    def test_detects_feature_selection(self):
        pipe = create_ml_pipeline(feature_selection=True, n_features=3)
        info = get_ml_pipeline_info(pipe)
        assert info["has_feature_selection"] is True

    def test_detects_no_feature_selection(self):
        pipe = create_ml_pipeline(feature_selection=False)
        info = get_ml_pipeline_info(pipe)
        assert info["has_feature_selection"] is False

    def test_detects_scaling(self):
        pipe = create_ml_pipeline(scaling=True)
        info = get_ml_pipeline_info(pipe)
        assert info["has_scaling"] is True

    def test_detects_no_scaling(self):
        pipe = create_ml_pipeline(scaling=False)
        info = get_ml_pipeline_info(pipe)
        assert info["has_scaling"] is False


# ---------------------------------------------------------------------------
# optimize_ml_pipeline
# ---------------------------------------------------------------------------

class TestOptimizeMlPipeline:
    def test_regression_optimization(self, sample_xy):
        X, y = sample_xy
        pipe = optimize_ml_pipeline(X, y, task_type="regression")
        assert isinstance(pipe, Pipeline)

    def test_classification_optimization(self, sample_xy):
        X, y = sample_xy
        pipe = optimize_ml_pipeline(X, y, task_type="classification")
        assert isinstance(pipe, Pipeline)

    def test_unknown_task_type(self, sample_xy):
        X, y = sample_xy
        pipe = optimize_ml_pipeline(X, y, task_type="other")
        assert isinstance(pipe, Pipeline)

    def test_with_max_features(self, sample_xy):
        X, y = sample_xy
        pipe = optimize_ml_pipeline(X, y, task_type="regression", max_features=3)
        info = get_ml_pipeline_info(pipe)
        assert info["has_feature_selection"] is True
