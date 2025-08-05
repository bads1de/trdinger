"""
AlgorithmRegistryクラスのテスト
"""

import sys
import os
import pytest

# バックエンドディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from app.services.ml.common.algorithm_registry import AlgorithmRegistry


class TestAlgorithmRegistry:
    """AlgorithmRegistryクラスのテスト"""

    def test_get_algorithm_name(self):
        """クラス名からアルゴリズム名を取得するテスト"""
        # 既知のクラス名のテスト
        assert AlgorithmRegistry.get_algorithm_name("LightGBMModel") == "lightgbm"
        assert AlgorithmRegistry.get_algorithm_name("XGBoostModel") == "xgboost"
        assert AlgorithmRegistry.get_algorithm_name("CatBoostModel") == "catboost"
        assert AlgorithmRegistry.get_algorithm_name("RandomForestModel") == "randomforest"
        assert AlgorithmRegistry.get_algorithm_name("ExtraTreesModel") == "extratrees"
        assert AlgorithmRegistry.get_algorithm_name("AdaBoostModel") == "adaboost"
        assert AlgorithmRegistry.get_algorithm_name("GradientBoostingModel") == "gradientboosting"
        assert AlgorithmRegistry.get_algorithm_name("KNNModel") == "knn"
        assert AlgorithmRegistry.get_algorithm_name("NaiveBayesModel") == "naivebayes"
        assert AlgorithmRegistry.get_algorithm_name("RidgeModel") == "ridge"
        assert AlgorithmRegistry.get_algorithm_name("TabNetModel") == "tabnet"

        # 未知のクラス名のテスト
        assert AlgorithmRegistry.get_algorithm_name("UnknownModel") == "unknown"
        assert AlgorithmRegistry.get_algorithm_name("") == "unknown"

    def test_get_display_name(self):
        """アルゴリズム名から表示名を取得するテスト"""
        assert AlgorithmRegistry.get_display_name("lightgbm") == "LightGBM"
        assert AlgorithmRegistry.get_display_name("xgboost") == "XGBoost"
        assert AlgorithmRegistry.get_display_name("catboost") == "CatBoost"
        assert AlgorithmRegistry.get_display_name("randomforest") == "Random Forest"
        assert AlgorithmRegistry.get_display_name("extratrees") == "Extra Trees"
        assert AlgorithmRegistry.get_display_name("adaboost") == "AdaBoost"
        assert AlgorithmRegistry.get_display_name("gradientboosting") == "Gradient Boosting"
        assert AlgorithmRegistry.get_display_name("knn") == "K-Nearest Neighbors"
        assert AlgorithmRegistry.get_display_name("naivebayes") == "Naive Bayes"
        assert AlgorithmRegistry.get_display_name("ridge") == "Ridge Classifier"
        assert AlgorithmRegistry.get_display_name("tabnet") == "TabNet"

        # 未知のアルゴリズム名のテスト
        assert AlgorithmRegistry.get_display_name("unknown") == "Unknown"

    def test_is_supported_algorithm(self):
        """アルゴリズムがサポートされているかどうかを確認するテスト"""
        # サポートされているアルゴリズム
        assert AlgorithmRegistry.is_supported_algorithm("lightgbm") is True
        assert AlgorithmRegistry.is_supported_algorithm("xgboost") is True
        assert AlgorithmRegistry.is_supported_algorithm("catboost") is True
        assert AlgorithmRegistry.is_supported_algorithm("randomforest") is True
        assert AlgorithmRegistry.is_supported_algorithm("extratrees") is True
        assert AlgorithmRegistry.is_supported_algorithm("adaboost") is True
        assert AlgorithmRegistry.is_supported_algorithm("gradientboosting") is True
        assert AlgorithmRegistry.is_supported_algorithm("knn") is True
        assert AlgorithmRegistry.is_supported_algorithm("naivebayes") is True
        assert AlgorithmRegistry.is_supported_algorithm("ridge") is True
        assert AlgorithmRegistry.is_supported_algorithm("tabnet") is True

        # サポートされていないアルゴリズム
        assert AlgorithmRegistry.is_supported_algorithm("unknown") is False
        assert AlgorithmRegistry.is_supported_algorithm("") is False

    def test_get_supported_algorithms(self):
        """サポートされているアルゴリズムのリストを取得するテスト"""
        supported_algorithms = AlgorithmRegistry.get_supported_algorithms()
        
        # リストが正しい形式であることを確認
        assert isinstance(supported_algorithms, list)
        assert len(supported_algorithms) > 0
        
        # すべてのアルゴリズムがサポートされていることを確認
        for algorithm in supported_algorithms:
            assert AlgorithmRegistry.is_supported_algorithm(algorithm) is True
        
        # 期待されるアルゴリズムが含まれていることを確認
        expected_algorithms = [
            "lightgbm", "xgboost", "catboost", "randomforest", "extratrees",
            "adaboost", "gradientboosting", "knn", "naivebayes", "ridge", "tabnet"
        ]
        for algorithm in expected_algorithms:
            assert algorithm in supported_algorithms

    def test_algorithm_name_consistency(self):
        """アルゴリズム名の一貫性をテスト"""
        # すべてのサポートされているアルゴリズムについて、
        # 表示名が正しく取得できることを確認
        for algorithm in AlgorithmRegistry.get_supported_algorithms():
            display_name = AlgorithmRegistry.get_display_name(algorithm)
            assert display_name is not None
            assert isinstance(display_name, str)
            assert len(display_name) > 0