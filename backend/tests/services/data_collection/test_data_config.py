"""
Bybitデータ設定クラスのテストモジュール

DataServiceConfigデータクラスとファクトリ関数をテストします:
- DataServiceConfig のデフォルト値
- get_funding_rate_config
- get_open_interest_config
"""

import pytest

from app.services.data_collection.bybit.data_config import (
    DataServiceConfig,
    get_funding_rate_config,
    get_open_interest_config,
)


class TestDataServiceConfig:
    """DataServiceConfig データクラスのテスト"""

    def test_dataclass_creation(self):
        """必要なフィールドでインスタンスが作成される"""
        config = DataServiceConfig(
            repository_class=object,
            get_timestamp_method_name="get_ts",
            data_converter_class=object,
            converter_method_name="convert",
            fetch_history_method_name="fetch_history",
            fetch_current_method_name="fetch_current",
        )
        assert config.repository_class is object
        assert config.pagination_strategy == "until"
        assert config.default_limit == 100
        assert config.page_limit == 200
        assert config.max_pages == 50
        assert config.insert_method_name == "insert_data"
        assert config.log_prefix == "DATA"

    def test_custom_values(self):
        """カスタム値が正しく設定される"""
        config = DataServiceConfig(
            repository_class=str,
            get_timestamp_method_name="ts",
            data_converter_class=str,
            converter_method_name="conv",
            fetch_history_method_name="fh",
            fetch_current_method_name="fc",
            pagination_strategy="time_range",
            default_limit=200,
            page_limit=500,
            max_pages=100,
            insert_method_name="insert_custom",
            log_prefix="CUSTOM",
        )
        assert config.pagination_strategy == "time_range"
        assert config.default_limit == 200
        assert config.max_pages == 100
        assert config.log_prefix == "CUSTOM"


class TestGetFundingRateConfig:
    """get_funding_rate_config のテスト"""

    def test_returns_data_service_config(self):
        config = get_funding_rate_config()
        assert isinstance(config, DataServiceConfig)

    def test_funding_rate_specific_values(self):
        config = get_funding_rate_config()
        assert config.log_prefix == "FR"
        assert config.pagination_strategy == "until"
        assert config.insert_method_name == "insert_funding_rate_data"
        assert config.get_timestamp_method_name == "get_latest_funding_timestamp"
        assert config.converter_method_name == "ccxt_to_db_format"
        assert config.default_limit == 100
        assert config.page_limit == 200
        assert config.max_pages == 50

    def test_funding_rate_config_has_correct_method_names(self):
        config = get_funding_rate_config()
        assert config.fetch_history_method_name == "fetch_funding_rate_history"
        assert config.fetch_current_method_name == "fetch_funding_rate"


class TestGetOpenInterestConfig:
    """get_open_interest_config のテスト"""

    def test_returns_data_service_config(self):
        config = get_open_interest_config()
        assert isinstance(config, DataServiceConfig)

    def test_open_interest_specific_values(self):
        config = get_open_interest_config()
        assert config.log_prefix == "OI"
        assert config.pagination_strategy == "time_range"
        assert config.insert_method_name == "insert_open_interest_data"
        assert config.get_timestamp_method_name == "get_latest_open_interest_timestamp"
        assert config.converter_method_name == "ccxt_to_db_format"
        assert config.max_pages == 500  # FR より多い

    def test_open_interest_config_has_correct_method_names(self):
        config = get_open_interest_config()
        assert config.fetch_history_method_name == "fetch_open_interest_history"
        assert config.fetch_current_method_name == "fetch_open_interest"

    def test_configs_have_different_pagination_strategies(self):
        """FR と OI で pagination_strategy が異なることを確認"""
        fr_config = get_funding_rate_config()
        oi_config = get_open_interest_config()
        assert fr_config.pagination_strategy != oi_config.pagination_strategy
        assert fr_config.pagination_strategy == "until"
        assert oi_config.pagination_strategy == "time_range"

    def test_configs_have_different_max_pages(self):
        """FR と OI で max_pages が異なることを確認"""
        fr_config = get_funding_rate_config()
        oi_config = get_open_interest_config()
        assert oi_config.max_pages > fr_config.max_pages
