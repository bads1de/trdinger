"""
Feature Selection Strategies Base クラスのテスト
"""

import numpy as np
import pytest

from app.services.ml.feature_selection.strategies.base import BaseSelectionStrategy
from app.services.ml.feature_selection.config import FeatureSelectionConfig


class ConcreteSelectionStrategy(BaseSelectionStrategy):
    """テスト用の具象クラス"""

    def select(self, X, y, feature_names, config):
        """全ての特徴量を選択する"""
        mask = np.ones(len(feature_names), dtype=bool)
        scores = np.ones(len(feature_names))
        return mask, {"scores": scores}


class TestBaseSelectionStrategy:
    """BaseSelectionStrategyのテスト"""

    def test_cannot_instantiate_abstract_class(self):
        """抽象クラスは直接インスタンス化できないこと"""
        with pytest.raises(TypeError):
            BaseSelectionStrategy()

    def test_concrete_class_can_instantiate(self):
        """具象クラスはインスタンス化できること"""
        strategy = ConcreteSelectionStrategy()
        assert strategy is not None

    def test_select_method(self):
        """selectメソッドが動作すること"""
        strategy = ConcreteSelectionStrategy()
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        config = FeatureSelectionConfig()

        mask, details = strategy.select(X, y, feature_names, config)
        assert len(mask) == 5
        assert mask.all()
        assert "scores" in details


class TestLimitFeatures:
    """_limit_featuresメソッドのテスト"""

    def test_no_limit_when_max_features_is_none(self):
        """max_featuresがNoneの場合、制限なし"""
        strategy = ConcreteSelectionStrategy()
        config = FeatureSelectionConfig(max_features=None)
        mask = np.array([True, True, True, False, False])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        result = strategy._limit_features(mask, scores, config)
        np.testing.assert_array_equal(result, mask)

    def test_no_limit_when_within_max_features(self):
        """選択数がmax_features以下の場合、制限なし"""
        strategy = ConcreteSelectionStrategy()
        config = FeatureSelectionConfig(max_features=5)
        mask = np.array([True, True, True, False, False])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        result = strategy._limit_features(mask, scores, config)
        np.testing.assert_array_equal(result, mask)

    def test_limit_applied_when_exceeds_max_features(self):
        """選択数がmax_featuresを超える場合、制限が適用されること"""
        strategy = ConcreteSelectionStrategy()
        config = FeatureSelectionConfig(max_features=2)
        mask = np.array([True, True, True, True, False])
        scores = np.array([0.5, 0.9, 0.3, 0.8, 0.1])

        result = strategy._limit_features(mask, scores, config)
        assert result.sum() == 2
        # スコアが高い特徴量（f2: 0.9, f4: 0.8）が選択されるはず
        assert result[1] == True  # f2
        assert result[3] == True  # f4

    def test_limit_preserves_highest_scores(self):
        """最高スコアの特徴量が保持されること"""
        strategy = ConcreteSelectionStrategy()
        config = FeatureSelectionConfig(max_features=3)
        mask = np.array([True, True, True, True, True])
        scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])

        result = strategy._limit_features(mask, scores, config)
        assert result.sum() == 3
        # スコアの高い順: f3(0.9), f5(0.7), f2(0.5)
        assert result[2] == True  # f3
        assert result[4] == True  # f5
        assert result[1] == True  # f2
