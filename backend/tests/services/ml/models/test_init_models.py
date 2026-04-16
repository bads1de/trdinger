"""
ML Models パッケージのテスト
"""

import pytest

from app.services.ml.models import (
    CatBoostModel,
    LightGBMModel,
    XGBoostModel,
    algorithm_registry,
    get_available_models,
)


class TestModelsInit:
    """モデルパッケージのインポートテスト"""

    def test_lightgbm_model_import(self):
        """LightGBMModelがインポートできること"""
        assert LightGBMModel is not None

    def test_xgboost_model_import(self):
        """XGBoostModelがインポートできること"""
        assert XGBoostModel is not None

    def test_catboost_model_import(self):
        """CatBoostModelがインポートできること"""
        assert CatBoostModel is not None

    def test_algorithm_registry_import(self):
        """algorithm_registryがインポートできること"""
        assert algorithm_registry is not None


class TestGetAvailableModels:
    """get_available_models関数のテスト"""

    def test_returns_list(self):
        """リストが返されること"""
        result = get_available_models()
        assert isinstance(result, list)

    def test_contains_expected_models(self):
        """期待されるモデルが含まれること"""
        result = get_available_models()
        # 少なくとも1つのモデルが利用可能なはず
        assert len(result) > 0
        # lightgbmは通常利用可能
        assert "lightgbm" in result
