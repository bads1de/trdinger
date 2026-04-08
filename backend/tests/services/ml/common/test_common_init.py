"""
ml/commonパッケージの__init__.pyのテスト

遅延ロード機能（__getattr__）とエクスポート定義を確認します。
"""

import pytest

import app.services.ml.common as common_package


class TestMLCommonInitExports:
    """ml/common/__init__.pyのエクスポートテスト"""

    def test_base_resource_manager_exported(self):
        """BaseResourceManagerがエクスポートされている"""
        assert hasattr(common_package, "BaseResourceManager")

    def test_cleanup_level_exported(self):
        """CleanupLevelがエクスポートされている"""
        assert hasattr(common_package, "CleanupLevel")

    def test_ml_base_error_exported(self):
        """MLBaseErrorがエクスポートされている"""
        assert hasattr(common_package, "MLBaseError")

    def test_ml_data_error_exported(self):
        """MLDataErrorがエクスポートされている"""
        assert hasattr(common_package, "MLDataError")

    def test_ml_feature_error_exported(self):
        """MLFeatureErrorがエクスポートされている"""
        assert hasattr(common_package, "MLFeatureError")

    def test_ml_model_error_exported(self):
        """MLModelErrorがエクスポートされている"""
        assert hasattr(common_package, "MLModelError")

    def test_ml_prediction_error_exported(self):
        """MLPredictionErrorがエクスポートされている"""
        assert hasattr(common_package, "MLPredictionError")

    def test_ml_training_error_exported(self):
        """MLTrainingErrorがエクスポートされている"""
        assert hasattr(common_package, "MLTrainingError")

    def test_ml_validation_error_exported(self):
        """MLValidationErrorがエクスポートされている"""
        assert hasattr(common_package, "MLValidationError")

    def test_algorithm_registry_exported(self):
        """AlgorithmRegistryがエクスポートされている"""
        assert hasattr(common_package, "AlgorithmRegistry")

    def test_algorithm_registry_instance_exported(self):
        """algorithm_registryインスタンスがエクスポートされている"""
        assert hasattr(common_package, "algorithm_registry")

    def test_model_metadata_exported(self):
        """ModelMetadataがエクスポートされている"""
        assert hasattr(common_package, "ModelMetadata")

    def test_optimize_dtypes_exported(self):
        """optimize_dtypesがエクスポートされている"""
        assert hasattr(common_package, "optimize_dtypes")

    def test_generate_cache_key_exported(self):
        """generate_cache_keyがエクスポートされている"""
        assert hasattr(common_package, "generate_cache_key")

    def test_validate_training_inputs_exported(self):
        """validate_training_inputsがエクスポートされている"""
        assert hasattr(common_package, "validate_training_inputs")

    def test_prepare_data_for_prediction_exported(self):
        """prepare_data_for_predictionがエクスポートされている"""
        assert hasattr(common_package, "prepare_data_for_prediction")

    def test_predict_class_from_proba_exported(self):
        """predict_class_from_probaがエクスポートされている"""
        assert hasattr(common_package, "predict_class_from_proba")

    def test_get_feature_importance_unified_exported(self):
        """get_feature_importance_unifiedがエクスポートされている"""
        assert hasattr(common_package, "get_feature_importance_unified")

    def test_calculate_price_change_exported(self):
        """calculate_price_changeがエクスポートされている"""
        assert hasattr(common_package, "calculate_price_change")

    def test_calculate_volatility_std_exported(self):
        """calculate_volatility_stdがエクスポートされている"""
        assert hasattr(common_package, "calculate_volatility_std")

    def test_calculate_volatility_atr_exported(self):
        """calculate_volatility_atrがエクスポートされている"""
        assert hasattr(common_package, "calculate_volatility_atr")

    def test_calculate_historical_volatility_exported(self):
        """calculate_historical_volatilityがエクスポートされている"""
        assert hasattr(common_package, "calculate_historical_volatility")

    def test_calculate_realized_volatility_exported(self):
        """calculate_realized_volatilityがエクスポートされている"""
        assert hasattr(common_package, "calculate_realized_volatility")

    def test_ml_config_manager_lazy_load(self):
        """MLConfigManagerが遅延ロードされる"""
        from app.services.ml.common.config import MLConfigManager

        manager = getattr(common_package, "MLConfigManager")

        assert manager is MLConfigManager

    def test_ml_config_manager_instance_lazy_load(self):
        """ml_config_managerインスタンスが遅延ロードされる"""
        from app.services.ml.common.config import ml_config_manager

        manager = getattr(common_package, "ml_config_manager")

        assert manager is ml_config_manager

    def test_get_default_ensemble_config_lazy_load(self):
        """get_default_ensemble_configが遅延ロードされる"""
        from app.services.ml.common.config import get_default_ensemble_config

        func = getattr(common_package, "get_default_ensemble_config")

        assert func is get_default_ensemble_config

    def test_get_default_single_model_config_lazy_load(self):
        """get_default_single_model_configが遅延ロードされる"""
        from app.services.ml.common.config import get_default_single_model_config

        func = getattr(common_package, "get_default_single_model_config")

        assert func is get_default_single_model_config

    def test_getattr_raises_for_non_existent(self):
        """存在しない属性でAttributeErrorが発生する"""
        with pytest.raises(AttributeError, match="module.*has no attribute"):
            _ = common_package.NonExistentAttribute

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "CleanupLevel",
            "BaseResourceManager",
            "MLBaseError",
            "MLDataError",
            "MLValidationError",
            "MLModelError",
            "MLTrainingError",
            "MLPredictionError",
            "MLFeatureError",
            "AlgorithmRegistry",
            "algorithm_registry",
            "ModelMetadata",
            "optimize_dtypes",
            "generate_cache_key",
            "validate_training_inputs",
            "prepare_data_for_prediction",
            "predict_class_from_proba",
            "get_feature_importance_unified",
            "calculate_price_change",
            "calculate_volatility_std",
            "calculate_volatility_atr",
            "calculate_historical_volatility",
            "calculate_realized_volatility",
            "MLConfigManager",
            "ml_config_manager",
            "get_default_ensemble_config",
            "get_default_single_model_config",
        ]

        for item in expected_items:
            assert item in common_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(common_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert common_package.__doc__ is not None
        assert len(common_package.__doc__) > 0
