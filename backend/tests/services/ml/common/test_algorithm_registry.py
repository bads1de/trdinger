import pytest
from app.services.ml.common.algorithm_registry import AlgorithmRegistry

class TestAlgorithmRegistry:
    def test_get_algorithm_name_mapping(self):
        """直接マッピングのテスト"""
        assert AlgorithmRegistry.get_algorithm_name("LightGBMModel") == "lightgbm"
        assert AlgorithmRegistry.get_algorithm_name("XGBClassifier") == "xgboost"
        assert AlgorithmRegistry.get_algorithm_name("StackingEnsemble") == "stacking"

    def test_get_algorithm_name_inference(self):
        """接尾辞除去による推測のテスト"""
        # 'lightgbm' はサポートされているので推測可能
        assert AlgorithmRegistry.get_algorithm_name("lightgbm") == "lightgbm"
        # 'custom' はサポートされていないので 'unknown'
        assert AlgorithmRegistry.get_algorithm_name("CustomModel") == "unknown"

    def test_get_algorithm_name_empty(self):
        """空文字やNoneの場合"""
        assert AlgorithmRegistry.get_algorithm_name("") == "unknown"
        assert AlgorithmRegistry.get_algorithm_name(None) == "unknown"
