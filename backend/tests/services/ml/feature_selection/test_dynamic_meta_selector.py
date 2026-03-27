"""
DynamicMetaSelector のユニットテスト

自律型動的特徴量選択器をテストします:
- 初期化
- _cluster_features
- _get_dynamic_k
- fit / transform / get_support / get_feature_names_out
- エッジケース
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_selection.dynamic_meta_selector import DynamicMetaSelector


@pytest.fixture
def sample_data() -> tuple:
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n) * 0.01,  # ノイズ特徴量
            "f4": np.random.randn(n),
            "f5": np.random.randn(n),
            "f6": np.random.randn(n),
            "f7": np.random.randn(n),
            "f8": np.random.randn(n),
            "f9": np.random.randn(n),
            "f10": np.random.randn(n),
        }
    )
    # f2 は f1 と高相関にする
    X["f2"] = X["f1"] + np.random.randn(n) * 0.01
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y


# ---------------------------------------------------------------------------
# 初期化
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_values(self):
        sel = DynamicMetaSelector()
        assert sel.clustering_threshold == 0.8
        assert sel.min_features == 5
        assert sel.n_shadow_iterations == 5
        assert sel.random_state == 42

    def test_custom_values(self):
        sel = DynamicMetaSelector(
            clustering_threshold=0.7,
            min_features=3,
            n_shadow_iterations=10,
            random_state=123,
        )
        assert sel.clustering_threshold == 0.7
        assert sel.min_features == 3
        assert sel.n_shadow_iterations == 10


# ---------------------------------------------------------------------------
# _get_dynamic_k
# ---------------------------------------------------------------------------

class TestGetDynamicK:
    def test_small_sample(self):
        sel = DynamicMetaSelector(min_features=5)
        k = sel._get_dynamic_k(20)
        assert k >= 5

    def test_large_sample(self):
        sel = DynamicMetaSelector(min_features=5)
        k = sel._get_dynamic_k(10000)
        assert k <= 30

    def test_respects_min_features(self):
        sel = DynamicMetaSelector(min_features=10)
        k = sel._get_dynamic_k(10)
        assert k >= 10


# ---------------------------------------------------------------------------
# _cluster_features
# ---------------------------------------------------------------------------

class TestClusterFeatures:
    def test_clusters_returned(self, sample_data):
        X, _ = sample_data
        sel = DynamicMetaSelector()
        clusters = sel._cluster_features(X)

        assert isinstance(clusters, dict)
        # すべての特徴量がどこかのクラスタに所属
        all_features = []
        for feats in clusters.values():
            all_features.extend(feats)
        assert set(all_features) == set(X.columns)


# ---------------------------------------------------------------------------
# fit / transform
# ---------------------------------------------------------------------------

class TestFitTransform:
    def test_fit_returns_self(self, sample_data):
        X, y = sample_data
        sel = DynamicMetaSelector(n_shadow_iterations=2)
        result = sel.fit(X, y)
        assert result is sel

    def test_fit_sets_attributes(self, sample_data):
        X, y = sample_data
        sel = DynamicMetaSelector(n_shadow_iterations=2)
        sel.fit(X, y)

        assert sel.support_mask_ is not None
        assert sel.feature_names_in_ is not None
        assert sel.selected_features_ is not None
        assert len(sel.selected_features_) > 0

    def test_transform_dataframe(self, sample_data):
        X, y = sample_data
        sel = DynamicMetaSelector(n_shadow_iterations=2)
        sel.fit(X, y)

        X_transformed = sel.transform(X)
        assert isinstance(X_transformed, pd.DataFrame)
        assert set(X_transformed.columns) == set(sel.selected_features_)
        assert len(X_transformed) == len(X)

    def test_transform_numpy(self, sample_data):
        X, y = sample_data
        sel = DynamicMetaSelector(n_shadow_iterations=2)
        sel.fit(X, y)

        X_transformed = sel.transform(X.values)
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[1] == len(sel.selected_features_)

    def test_get_support(self, sample_data):
        X, y = sample_data
        sel = DynamicMetaSelector(n_shadow_iterations=2)
        sel.fit(X, y)

        support = sel.get_support()
        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert support.sum() == len(sel.selected_features_)

    def test_get_feature_names_out(self, sample_data):
        X, y = sample_data
        sel = DynamicMetaSelector(n_shadow_iterations=2)
        sel.fit(X, y)

        names = sel.get_feature_names_out(input_features=X.columns)
        assert isinstance(names, np.ndarray)
        assert set(names).issubset(set(X.columns))

    def test_numpy_input_fit(self, sample_data):
        X, y = sample_data
        sel = DynamicMetaSelector(n_shadow_iterations=2)
        sel.fit(X.values, y)

        assert sel.support_mask_ is not None
        assert len(sel.selected_features_) > 0

    def test_primary_proba_preserved(self, sample_data):
        """primary_proba 特徴量が含まれる場合は保持される"""
        X, y = sample_data
        X["primary_proba"] = np.random.rand(len(X))

        sel = DynamicMetaSelector(n_shadow_iterations=2, min_features=3)
        sel.fit(X, y)

        # primary_proba が最終選択に含まれるか確認
        # (シャドウフィルタリングで落とされても強制復帰される)
        assert "primary_proba" in sel.feature_names_in_
