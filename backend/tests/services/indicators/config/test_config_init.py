"""
indicators/configパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import app.services.indicators.config as config_package


class TestIndicatorsConfigInitExports:
    """indicators/config/__init__.pyのエクスポートテスト"""

    def test_indicator_config_exported(self):
        """IndicatorConfigがエクスポートされている"""
        assert hasattr(config_package, "IndicatorConfig")

    def test_parameter_config_exported(self):
        """ParameterConfigがエクスポートされている"""
        assert hasattr(config_package, "ParameterConfig")

    def test_indicator_result_type_exported(self):
        """IndicatorResultTypeがエクスポートされている"""
        assert hasattr(config_package, "IndicatorResultType")

    def test_indicator_scale_type_exported(self):
        """IndicatorScaleTypeがエクスポートされている"""
        assert hasattr(config_package, "IndicatorScaleType")

    def test_indicator_config_registry_exported(self):
        """IndicatorConfigRegistryがエクスポートされている"""
        assert hasattr(config_package, "IndicatorConfigRegistry")

    def test_indicator_registry_exported(self):
        """indicator_registryがエクスポートされている"""
        assert hasattr(config_package, "indicator_registry")

    def test_initialize_all_indicators_exported(self):
        """initialize_all_indicatorsがエクスポートされている"""
        assert hasattr(config_package, "initialize_all_indicators")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "IndicatorConfig",
            "ParameterConfig",
            "IndicatorResultType",
            "IndicatorScaleType",
            "IndicatorConfigRegistry",
            "indicator_registry",
            "initialize_all_indicators",
        ]

        for item in expected_items:
            assert item in config_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(config_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert config_package.__doc__ is not None
        assert len(config_package.__doc__) > 0
