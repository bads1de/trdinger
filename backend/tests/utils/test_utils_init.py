"""
utilsパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import app.utils as utils_package


class TestUtilsInitExports:
    """utils/__init__.pyのエクスポートテスト"""

    def test_response_utilities_exported(self):
        """レスポンスユーティリティがエクスポートされている"""
        assert hasattr(utils_package, "api_response")
        assert hasattr(utils_package, "error_response")

    def test_error_handler_exported(self):
        """エラーハンドラーがエクスポートされている"""
        assert hasattr(utils_package, "ErrorHandler")
        assert hasattr(utils_package, "safe_execute")
        assert hasattr(utils_package, "safe_operation")
        assert hasattr(utils_package, "operation_context")
        assert hasattr(utils_package, "get_memory_usage_mb")

    def test_custom_exceptions_exported(self):
        """カスタム例外がエクスポートされている"""
        assert hasattr(utils_package, "TimeoutError")
        assert hasattr(utils_package, "ValidationError")
        assert hasattr(utils_package, "DataError")
        assert hasattr(utils_package, "ModelError")

    def test_data_converters_exported(self):
        """データコンバーターがエクスポートされている"""
        assert hasattr(utils_package, "OHLCVDataConverter")
        assert hasattr(utils_package, "FundingRateDataConverter")
        assert hasattr(utils_package, "OpenInterestDataConverter")
        assert hasattr(utils_package, "DataConversionError")
        assert hasattr(utils_package, "parse_timestamp_safe")

    def test_serialization_utilities_exported(self):
        """シリアライゼーションユーティリティがエクスポートされている"""
        assert hasattr(utils_package, "dataclass_to_dict")
        assert hasattr(utils_package, "dataclass_to_json")
        assert hasattr(utils_package, "dataclass_from_dict")
        assert hasattr(utils_package, "dataclass_from_json")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            # レスポンスユーティリティ
            "api_response",
            "error_response",
            # エラーハンドリング・モニタリング
            "ErrorHandler",
            "safe_execute",
            "safe_operation",
            "operation_context",
            "get_memory_usage_mb",
            # カスタム例外
            "TimeoutError",
            "ValidationError",
            "DataError",
            "ModelError",
            # データ変換
            "OHLCVDataConverter",
            "FundingRateDataConverter",
            "OpenInterestDataConverter",
            "DataConversionError",
            "parse_timestamp_safe",
            # シリアライズ
            "dataclass_to_dict",
            "dataclass_to_json",
            "dataclass_from_dict",
            "dataclass_from_json",
        ]

        for item in expected_items:
            assert item in utils_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(utils_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert utils_package.__doc__ is not None
        assert len(utils_package.__doc__) > 0
