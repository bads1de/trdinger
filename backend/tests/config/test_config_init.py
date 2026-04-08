"""
configパッケージの__init__.pyのテスト

遅延ロード機能（__getattr__, __dir__）をテストします。
"""

import pytest

import app.config as config_package


class TestConfigInitExports:
    """config/__init__.pyのエクスポートテスト"""

    def test_constants_exported(self):
        """共通定数がエクスポートされている"""
        assert hasattr(config_package, "SUPPORTED_TIMEFRAMES")
        assert hasattr(config_package, "DEFAULT_ENSEMBLE_ALGORITHMS")
        assert hasattr(config_package, "DEFAULT_MARKET_EXCHANGE")
        assert hasattr(config_package, "DEFAULT_MARKET_SYMBOL")
        assert hasattr(config_package, "DEFAULT_MARKET_TIMEFRAME")
        assert hasattr(config_package, "DEFAULT_DATA_LIMIT")
        assert hasattr(config_package, "MAX_DATA_LIMIT")
        assert hasattr(config_package, "MIN_DATA_LIMIT")

    def test_constants_values(self):
        """定数の値が正しい"""
        assert config_package.DEFAULT_MARKET_EXCHANGE == "bybit"
        assert config_package.DEFAULT_MARKET_SYMBOL == "BTC/USDT:USDT"
        assert config_package.DEFAULT_MARKET_TIMEFRAME == "1h"
        assert config_package.DEFAULT_DATA_LIMIT == 100
        assert config_package.MAX_DATA_LIMIT == 1000
        assert config_package.MIN_DATA_LIMIT == 1


class TestLazyLoadGetAttr:
    """__getattr__による遅延ロードのテスト"""

    def test_unified_config_lazy_load(self):
        """UnifiedConfigが遅延ロードされる"""
        from app.config.unified_config import UnifiedConfig

        # __getattr__経由でアクセス
        unified_config = getattr(config_package, "UnifiedConfig")

        assert unified_config is UnifiedConfig

    def test_app_config_lazy_load(self):
        """AppConfigが遅延ロードされる"""
        from app.config.unified_config import AppConfig

        app_config = getattr(config_package, "AppConfig")

        assert app_config is AppConfig

    def test_database_config_lazy_load(self):
        """DatabaseConfigが遅延ロードされる"""
        from app.config.unified_config import DatabaseConfig

        db_config = getattr(config_package, "DatabaseConfig")

        assert db_config is DatabaseConfig

    def test_logging_config_lazy_load(self):
        """LoggingConfigが遅延ロードされる"""
        from app.config.unified_config import LoggingConfig

        logging_config = getattr(config_package, "LoggingConfig")

        assert logging_config is LoggingConfig

    def test_market_config_lazy_load(self):
        """MarketConfigが遅延ロードされる"""
        from app.config.unified_config import MarketConfig

        market_config = getattr(config_package, "MarketConfig")

        assert market_config is MarketConfig

    def test_data_collection_config_lazy_load(self):
        """DataCollectionConfigが遅延ロードされる"""
        from app.config.unified_config import DataCollectionConfig

        data_collection_config = getattr(config_package, "DataCollectionConfig")

        assert data_collection_config is DataCollectionConfig

    def test_getattr_raises_for_non_existent(self):
        """存在しない属性でAttributeErrorが発生する"""
        with pytest.raises(AttributeError, match="module.*has no attribute"):
            _ = config_package.NonExistentAttribute


class TestLazyLoadDir:
    """__dir__のテスト"""

    def test_dir_includes_config_classes(self):
        """__dir__に設定クラスが含まれる"""
        dir_result = dir(config_package)

        assert "UnifiedConfig" in dir_result
        assert "AppConfig" in dir_result
        assert "DatabaseConfig" in dir_result
        assert "LoggingConfig" in dir_result
        assert "MarketConfig" in dir_result
        assert "DataCollectionConfig" in dir_result

    def test_dir_includes_constants(self):
        """__dir__に定数が含まれる"""
        dir_result = dir(config_package)

        assert "SUPPORTED_TIMEFRAMES" in dir_result
        assert "DEFAULT_MARKET_EXCHANGE" in dir_result
        assert "DEFAULT_DATA_LIMIT" in dir_result

    def test_dir_returns_list(self):
        """__dir__がリストを返す"""
        dir_result = dir(config_package)

        assert isinstance(dir_result, list)
        assert len(dir_result) > 0


class TestConfigInitAll:
    """__all__定義のテスト"""

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "SUPPORTED_TIMEFRAMES",
            "DEFAULT_ENSEMBLE_ALGORITHMS",
            "DEFAULT_MARKET_EXCHANGE",
            "DEFAULT_MARKET_SYMBOL",
            "DEFAULT_MARKET_TIMEFRAME",
            "DEFAULT_DATA_LIMIT",
            "MAX_DATA_LIMIT",
            "MIN_DATA_LIMIT",
            "UnifiedConfig",
            "AppConfig",
            "DatabaseConfig",
            "LoggingConfig",
            "MarketConfig",
            "DataCollectionConfig",
        ]

        for item in expected_items:
            assert item in config_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(config_package.__all__, list)


class TestConfigModuleAttributes:
    """configモジュールの属性テスト"""

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert config_package.__doc__ is not None
        assert len(config_package.__doc__) > 0

    def test_module_name(self):
        """モジュール名が正しい"""
        assert config_package.__name__ == "app.config"
