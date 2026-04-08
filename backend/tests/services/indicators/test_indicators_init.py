"""
indicatorsパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import app.services.indicators as indicators_package


class TestIndicatorsInitExports:
    """indicators/__init__.pyのエクスポートテスト"""

    def test_trend_indicators_exported(self):
        """TrendIndicatorsがエクスポートされている"""
        assert hasattr(indicators_package, "TrendIndicators")

    def test_momentum_indicators_exported(self):
        """MomentumIndicatorsがエクスポートされている"""
        assert hasattr(indicators_package, "MomentumIndicators")

    def test_volatility_indicators_exported(self):
        """VolatilityIndicatorsがエクスポートされている"""
        assert hasattr(indicators_package, "VolatilityIndicators")

    def test_volume_indicators_exported(self):
        """VolumeIndicatorsがエクスポートされている"""
        assert hasattr(indicators_package, "VolumeIndicators")

    def test_original_indicators_exported(self):
        """OriginalIndicatorsがエクスポートされている"""
        assert hasattr(indicators_package, "OriginalIndicators")

    def test_pandas_ta_error_exported(self):
        """PandasTAErrorがエクスポートされている"""
        assert hasattr(indicators_package, "PandasTAError")

    def test_validate_input_exported(self):
        """validate_inputがエクスポートされている"""
        assert hasattr(indicators_package, "validate_input")

    def test_validate_data_length_with_fallback_exported(self):
        """validate_data_length_with_fallbackがエクスポートされている"""
        assert hasattr(indicators_package, "validate_data_length_with_fallback")

    def test_validate_series_params_exported(self):
        """validate_series_paramsがエクスポートされている"""
        assert hasattr(indicators_package, "validate_series_params")

    def test_validate_multi_series_params_exported(self):
        """validate_multi_series_paramsがエクスポートされている"""
        assert hasattr(indicators_package, "validate_multi_series_params")

    def test_handle_pandas_ta_errors_exported(self):
        """handle_pandas_ta_errorsがエクスポートされている"""
        assert hasattr(indicators_package, "handle_pandas_ta_errors")

    def test_technical_indicator_service_exported(self):
        """TechnicalIndicatorServiceがエクスポートされている"""
        assert hasattr(indicators_package, "TechnicalIndicatorService")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "TrendIndicators",
            "MomentumIndicators",
            "VolatilityIndicators",
            "VolumeIndicators",
            "OriginalIndicators",
            "PandasTAError",
            "validate_input",
            "validate_data_length_with_fallback",
            "validate_series_params",
            "validate_multi_series_params",
            "handle_pandas_ta_errors",
            "TechnicalIndicatorService",
        ]

        for item in expected_items:
            assert item in indicators_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(indicators_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert indicators_package.__doc__ is not None
        assert len(indicators_package.__doc__) > 0
