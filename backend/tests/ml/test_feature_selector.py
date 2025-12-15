"""
FeatureSelector のユニットテスト
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from app.services.ml.feature_selection.feature_selector import (
    FeatureSelector,
    FeatureSelectionConfig,
    SelectionMethod,
)


@pytest.fixture
def sample_data():
    """テスト用のサンプルデータを生成"""
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    # 重要な特徴量（ターゲットと相関）
    X_important = np.random.randn(n_samples, 5)
    y = (X_important[:, 0] + X_important[:, 1] > 0).astype(int)

    # ノイズ特徴量
    X_noise = np.random.randn(n_samples, n_features - 5)

    X = pd.DataFrame(
        np.hstack([X_important, X_noise]),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(y, name="target")

    return X, y


class TestFeatureSelectorInit:
    """初期化のテスト"""

    def test_default_config(self):
        """デフォルト設定での初期化"""
        selector = FeatureSelector()
        assert selector.config.method == SelectionMethod.ENSEMBLE
        assert selector.config.ensemble_methods is not None
        assert len(selector.config.ensemble_methods) > 0

    def test_custom_config(self):
        """カスタム設定での初期化"""
        config = FeatureSelectionConfig(
            method=SelectionMethod.LASSO,
            k_features=10,
        )
        selector = FeatureSelector(config)
        assert selector.config.method == SelectionMethod.LASSO
        assert selector.config.k_features == 10


class TestLassoSelection:
    """Lasso選択のテスト"""

    def test_lasso_selection_returns_features(self, sample_data):
        """Lasso選択が特徴量を返すことを確認"""
        X, y = sample_data
        config = FeatureSelectionConfig(method=SelectionMethod.LASSO)
        selector = FeatureSelector(config)

        X_selected, results = selector.fit_transform(X, y)

        assert len(X_selected.columns) > 0
        assert "selected_features" in results
        assert results["method"] == "lasso"

    def test_lasso_uses_prefit_true(self, sample_data):
        """Lassoがprefit=Trueを使用することを確認"""
        X, y = sample_data
        config = FeatureSelectionConfig(method=SelectionMethod.LASSO)
        selector = FeatureSelector(config)

        with patch(
            "app.services.ml.feature_selection.feature_selector.SelectFromModel"
        ) as MockSelectFromModel:
            mock_instance = MagicMock()
            mock_instance.get_support.return_value = np.array([True] * 5 + [False] * 15)
            MockSelectFromModel.return_value = mock_instance

            selector.fit_transform(X, y)

            # SelectFromModelがprefit=Trueで呼ばれたことを確認
            MockSelectFromModel.assert_called_once()
            call_kwargs = MockSelectFromModel.call_args[1]
            assert (
                call_kwargs.get("prefit") is True
            ), "SelectFromModel should be called with prefit=True"


class TestRandomForestSelection:
    """ランダムフォレスト選択のテスト"""

    def test_random_forest_selection_returns_features(self, sample_data):
        """ランダムフォレスト選択が特徴量を返すことを確認"""
        X, y = sample_data
        config = FeatureSelectionConfig(method=SelectionMethod.RANDOM_FOREST)
        selector = FeatureSelector(config)

        X_selected, results = selector.fit_transform(X, y)

        assert len(X_selected.columns) > 0
        assert "selected_features" in results
        assert results["method"] == "random_forest"

    def test_random_forest_uses_prefit_true(self, sample_data):
        """ランダムフォレストがprefit=Trueを使用することを確認"""
        X, y = sample_data
        config = FeatureSelectionConfig(method=SelectionMethod.RANDOM_FOREST)
        selector = FeatureSelector(config)

        with patch(
            "app.services.ml.feature_selection.feature_selector.SelectFromModel"
        ) as MockSelectFromModel:
            mock_instance = MagicMock()
            mock_instance.get_support.return_value = np.array([True] * 5 + [False] * 15)
            MockSelectFromModel.return_value = mock_instance

            selector.fit_transform(X, y)

            # SelectFromModelがprefit=Trueで呼ばれたことを確認
            MockSelectFromModel.assert_called_once()
            call_kwargs = MockSelectFromModel.call_args[1]
            assert (
                call_kwargs.get("prefit") is True
            ), "SelectFromModel should be called with prefit=True"


class TestEnsembleSelection:
    """アンサンブル選択のテスト"""

    def test_ensemble_selection_returns_features(self, sample_data):
        """アンサンブル選択が特徴量を返すことを確認"""
        X, y = sample_data
        config = FeatureSelectionConfig(
            method=SelectionMethod.ENSEMBLE,
            ensemble_methods=[SelectionMethod.LASSO, SelectionMethod.RANDOM_FOREST],
        )
        selector = FeatureSelector(config)

        X_selected, results = selector.fit_transform(X, y)

        assert len(X_selected.columns) > 0
        assert results["method"] == "ensemble"
        assert "feature_votes" in results




