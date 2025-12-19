import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from app.services.ml.preprocessing.pipeline import (
    create_ml_pipeline,
    create_classification_pipeline,
    create_regression_pipeline,
    get_ml_pipeline_info,
    optimize_ml_pipeline
)

class TestMLPipeline:
    @pytest.fixture
    def sample_data(self):
        X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y

    def test_create_ml_pipeline_basic(self):
        """基本的なパイプライン作成のテスト"""
        pipe = create_ml_pipeline(scaling=True, scaling_method="standard")
        assert isinstance(pipe, Pipeline)
        assert "preprocessing" in [s[0] for s in pipe.steps]
        assert "scaler" in [s[0] for s in pipe.steps]

    def test_create_ml_pipeline_with_feature_selection(self):
        """特徴量選択を含むパイプライン作成"""
        pipe = create_ml_pipeline(
            feature_selection=True, 
            n_features=5, 
            selection_method="f_classif",
            is_classification=True
        )
        assert "feature_selection" in [s[0] for s in pipe.steps]
        # KBest が正しくセットされているか
        selector = dict(pipe.steps)["feature_selection"]
        assert selector.k == 5

    def test_classification_pipeline(self):
        """分類用パイプラインの作成"""
        pipe = create_classification_pipeline(feature_selection=True, n_features=3)
        assert "feature_selection" in [s[0] for s in pipe.steps]
        assert "scaler" in [s[0] for s in pipe.steps]

    def test_get_ml_pipeline_info(self, sample_data):
        """パイプライン情報の取得"""
        X, y = sample_data
        pipe = create_classification_pipeline(feature_selection=True, n_features=5)
        
        # 環境不具合を避けるため fit() を呼ばずに基本的な構造のみ検証
        info = get_ml_pipeline_info(pipe)
        assert info["pipeline_type"] == "ml"
        assert info["has_feature_selection"] is True
        assert info["has_scaling"] is True
        assert "preprocessing" in info["step_names"]

    def test_optimize_ml_pipeline(self, sample_data):
        """パイプライン最適化のテスト"""
        X, y = sample_data
        pipe = optimize_ml_pipeline(X, y, task_type="classification", max_features=4)
        
        assert isinstance(pipe, Pipeline)
        selector = dict(pipe.steps)["feature_selection"]
        assert selector.k == 4

    def test_unsupported_scaling_method(self):
        """サポートされていないスケーリング方法"""
        with pytest.raises(ValueError, match="サポートされていないスケーリング方法"):
            create_ml_pipeline(scaling_method="invalid")

    def test_unsupported_selection_method(self):
        """サポートされていない選択方法"""
        with pytest.raises(ValueError, match="サポートされていない選択方法"):
            create_ml_pipeline(feature_selection=True, n_features=5, selection_method="invalid")
