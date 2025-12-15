"""
Bybitファンディングレートサービスのテスト

BybitFundingRateServiceクラスの全機能をテストします:
- サービス初期化
- 現在のファンディングレート取得
- ファンディングレート履歴取得
- 全期間データ取得
- 差分更新
- データベース保存
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.data_collection.bybit.funding_rate_service import (
    BybitFundingRateService,
)


@pytest.fixture
def mock_exchange():
    """モックCCXT取引所"""
    exchange = MagicMock()
    exchange.fetch_funding_rate = AsyncMock()
    exchange.fetch_funding_rate_history = AsyncMock()
    return exchange


@pytest.fixture
def mock_config():
    """モック設定"""
    config = MagicMock()
    config.repository_class = MagicMock()
    config.get_timestamp_method_name = "get_latest_funding_timestamp"
    config.data_converter_class = MagicMock()
    config.converter_method_name = "ccxt_to_db_format"
    config.fetch_history_method_name = "fetch_funding_rate_history"
    config.fetch_current_method_name = "fetch_funding_rate"
    config.pagination_strategy = "until"
    config.default_limit = 100
    config.page_limit = 200
    config.max_pages = 50
    config.insert_method_name = "insert_funding_rate_data"
    config.log_prefix = "FR"
    return config


@pytest.fixture
def service(mock_exchange, mock_config):
    """サービスインスタンス"""
    with patch(
        "app.services.data_collection.bybit.bybit_service.ccxt.bybit"
    ) as mock_ccxt:
        mock_ccxt.return_value = mock_exchange
        with patch(
            "app.services.data_collection.bybit.funding_rate_service.get_funding_rate_config"
        ) as mock_get_config:
            mock_get_config.return_value = mock_config
            return BybitFundingRateService()


@pytest.fixture
def mock_repository():
    """モックリポジトリ"""
    repo = MagicMock()
    repo.get_latest_funding_timestamp = MagicMock()
    repo.insert_funding_rate_data = MagicMock(return_value=10)
    return repo


@pytest.mark.asyncio
class TestServiceInitialization:
    """サービス初期化テスト"""

    async def test_service_initialization(self, mock_exchange, mock_config):
        """サービスが正しく初期化されることを確認"""
        with patch(
            "app.services.data_collection.bybit.bybit_service.ccxt.bybit"
        ) as mock_ccxt:
            mock_ccxt.return_value = mock_exchange
            with patch(
                "app.services.data_collection.bybit.funding_rate_service.get_funding_rate_config"
            ) as mock_get_config:
                mock_get_config.return_value = mock_config

                service = BybitFundingRateService()

                assert service.exchange is not None
                assert service.config == mock_config
                mock_get_config.assert_called_once()

    async def test_config_loaded(self, service, mock_config):
        """設定が正しく読み込まれることを確認"""
        assert service.config == mock_config
        assert service.config.log_prefix == "FR"


@pytest.mark.asyncio
class TestParameterValidation:
    """パラメータ検証テスト"""

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_validate_parameters_valid(self, service):
        """有効なパラメータが検証を通過することを確認"""
        service._validate_parameters("BTC/USDT:USDT", 100)

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_validate_parameters_invalid_symbol(self, service):
        """無効なシンボルがValueErrorを発生させることを確認"""
        with pytest.raises(ValueError):
            service._validate_parameters("", 100)

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_validate_parameters_invalid_limit(self, service):
        """無効なlimitがValueErrorを発生させることを確認"""
        with pytest.raises(ValueError):
            service._validate_parameters("BTC/USDT:USDT", 0)


@pytest.mark.asyncio
class TestFetchCurrentFundingRate:
    """現在のファンディングレート取得テスト"""

    async def test_fetch_current_funding_rate_success(self, service, mock_exchange):
        """現在のファンディングレートが正常に取得できることを確認"""
        expected_data = {
            "symbol": "BTC/USDT:USDT",
            "fundingRate": 0.0001,
            "timestamp": 1609459200000,
        }

        mock_exchange.fetch_funding_rate.return_value = expected_data

        result = await service.fetch_current_funding_rate("BTC/USDT:USDT")

        assert result == expected_data

    async def test_fetch_current_funding_rate_with_normalized_symbol(
        self, service, mock_exchange
    ):
        """正規化されたシンボルで現在のファンディングレートが取得できることを確認"""
        expected_data = {
            "symbol": "BTC/USDT:USDT",
            "fundingRate": 0.0001,
            "timestamp": 1609459200000,
        }

        mock_exchange.fetch_funding_rate.return_value = expected_data

        result = await service.fetch_current_funding_rate("BTC/USDT:USDT")

        assert result == expected_data


@pytest.mark.asyncio
class TestFetchFundingRateHistory:
    """ファンディングレート履歴取得テスト"""

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_fetch_funding_rate_history_success(self, service, mock_exchange):
        """ファンディングレート履歴が正常に取得できることを確認"""
        expected_data = [
            {"timestamp": 1609459200000, "fundingRate": 0.0001},
            {"timestamp": 1609459300000, "fundingRate": 0.0002},
        ]

        mock_exchange.fetch_funding_rate_history.return_value = expected_data

        result = await service.fetch_funding_rate_history("BTC/USDT:USDT", limit=100)

        assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_fetch_funding_rate_history_with_since(self, service, mock_exchange):
        """sinceパラメータ付きでファンディングレート履歴が取得できることを確認"""
        since_timestamp = 1609459200000
        expected_data = [{"timestamp": 1609459300000, "fundingRate": 0.0002}]

        mock_exchange.fetch_funding_rate_history.return_value = expected_data

        result = await service.fetch_funding_rate_history(
            "BTC/USDT:USDT", limit=100, since=since_timestamp
        )

        assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_fetch_funding_rate_history_invalid_limit(self, service):
        """無効なlimitでValueErrorが発生することを確認"""
        with pytest.raises(ValueError):
            await service.fetch_funding_rate_history("BTC/USDT:USDT", limit=0)

    async def test_fetch_funding_rate_history_invalid_symbol(self, service):
        """無効なシンボルでValueErrorが発生することを確認"""
        with pytest.raises(ValueError):
            await service.fetch_funding_rate_history("", limit=100)


@pytest.mark.asyncio
class TestFetchAllFundingRateHistory:
    """全期間ファンディングレート取得テスト"""

    async def test_fetch_all_funding_rate_history_success(
        self, service, mock_exchange, mock_config
    ):
        """全期間のファンディングレートが取得できることを確認"""
        mock_data = [
            {"timestamp": 1609459200000, "fundingRate": 0.0001},
            {"timestamp": 1609459300000, "fundingRate": 0.0002},
        ]

        with patch.object(service, "_get_latest_timestamp_from_db", return_value=None):
            with patch.object(
                service, "_fetch_paginated_data", return_value=mock_data
            ) as mock_fetch:
                result = await service.fetch_all_funding_rate_history("BTC/USDT:USDT")

                assert result == mock_data
                mock_fetch.assert_called_once()

    async def test_fetch_all_funding_rate_history_with_existing_data(
        self, service, mock_exchange, mock_config
    ):
        """既存データありで全期間取得が実行されることを確認"""
        mock_data = [{"timestamp": 1609459300000, "fundingRate": 0.0002}]
        latest_timestamp = 1609459200000

        with patch.object(
            service, "_get_latest_timestamp_from_db", return_value=latest_timestamp
        ):
            with patch.object(
                service, "_fetch_paginated_data", return_value=mock_data
            ) as mock_fetch:
                result = await service.fetch_all_funding_rate_history("BTC/USDT:USDT")

                assert result == mock_data
                mock_fetch.assert_called_once()


@pytest.mark.asyncio
class TestFetchIncrementalData:
    """差分データ取得テスト"""

    async def test_fetch_incremental_funding_rate_data_no_existing(
        self, service, mock_repository, mock_config
    ):
        """既存データなしでの差分更新を確認"""
        mock_data = [
            {"timestamp": 1609459200000, "fundingRate": 0.0001},
            {"timestamp": 1609459300000, "fundingRate": 0.0002},
        ]

        mock_config.data_converter_class.ccxt_to_db_format = MagicMock(
            return_value=[{"id": 1}, {"id": 2}]
        )

        with patch.object(service, "_get_latest_timestamp_from_db", return_value=None):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = AsyncMock(return_value=mock_data)
                mock_loop.return_value.run_in_executor = mock_executor

                with patch(
                    "app.services.data_collection.bybit.bybit_service.get_db"
                ) as mock_get_db:
                    mock_db = MagicMock()
                    mock_get_db.return_value = iter([mock_db])
                    mock_repository_instance = MagicMock()
                    mock_repository_instance.insert_funding_rate_data = MagicMock(
                        return_value=2
                    )
                    mock_config.repository_class.return_value = mock_repository_instance

                    service.exchange.fetch_funding_rate_history.return_value = mock_data

                    result = await service.fetch_incremental_funding_rate_data(
                        "BTC/USDT:USDT"
                    )

                    assert result["success"] is True
                    assert result["saved_count"] == 2

    async def test_fetch_incremental_funding_rate_data_with_existing(
        self, service, mock_repository, mock_config
    ):
        """既存データありでの差分更新を確認"""
        latest_timestamp = 1609459200000
        mock_data = [{"timestamp": 1609459300000, "fundingRate": 0.0002}]

        mock_config.data_converter_class.ccxt_to_db_format = MagicMock(
            return_value=[{"id": 1}]
        )

        with patch.object(
            service, "_get_latest_timestamp_from_db", return_value=latest_timestamp
        ):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = AsyncMock(return_value=mock_data)
                mock_loop.return_value.run_in_executor = mock_executor

                with patch(
                    "app.services.data_collection.bybit.bybit_service.get_db"
                ) as mock_get_db:
                    mock_db = MagicMock()
                    mock_get_db.return_value = iter([mock_db])
                    mock_repository_instance = MagicMock()
                    mock_repository_instance.insert_funding_rate_data = MagicMock(
                        return_value=1
                    )
                    mock_config.repository_class.return_value = mock_repository_instance

                    service.exchange.fetch_funding_rate_history.return_value = mock_data

                    result = await service.fetch_incremental_funding_rate_data(
                        "BTC/USDT:USDT"
                    )

                    assert result["success"] is True
                    assert result["saved_count"] == 1
                    assert result["latest_timestamp"] == latest_timestamp

    async def test_fetch_incremental_with_custom_repository(
        self, service, mock_repository, mock_config
    ):
        """カスタムリポジトリでの差分更新を確認"""
        mock_data = [{"timestamp": 1609459200000, "fundingRate": 0.0001}]

        mock_config.data_converter_class.ccxt_to_db_format = MagicMock(
            return_value=[{"id": 1}]
        )
        mock_repository.insert_funding_rate_data = MagicMock(return_value=1)

        with patch.object(service, "_get_latest_timestamp_from_db", return_value=None):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = AsyncMock(return_value=mock_data)
                mock_loop.return_value.run_in_executor = mock_executor

                service.exchange.fetch_funding_rate_history.return_value = mock_data

                result = await service.fetch_incremental_funding_rate_data(
                    "BTC/USDT:USDT", mock_repository
                )

                assert result["success"] is True
                assert result["saved_count"] == 1


@pytest.mark.asyncio
class TestFetchAndSaveData:
    """データ取得・保存テスト"""

    async def test_fetch_and_save_funding_rate_data_success(
        self, service, mock_exchange, mock_config
    ):
        """ファンディングレートデータの取得・保存が成功することを確認"""
        mock_data = [
            {"timestamp": 1609459200000, "fundingRate": 0.0001},
            {"timestamp": 1609459300000, "fundingRate": 0.0002},
        ]

        mock_config.data_converter_class.ccxt_to_db_format = MagicMock(
            return_value=[{"id": 1}, {"id": 2}]
        )

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value=mock_data)
            mock_loop.return_value.run_in_executor = mock_executor

            with patch(
                "app.services.data_collection.bybit.bybit_service.get_db"
            ) as mock_get_db:
                mock_db = MagicMock()
                mock_get_db.return_value = iter([mock_db])
                mock_repository_instance = MagicMock()
                mock_repository_instance.insert_funding_rate_data = MagicMock(
                    return_value=2
                )
                mock_config.repository_class.return_value = mock_repository_instance

                service.exchange.fetch_funding_rate_history.return_value = mock_data

                result = await service.fetch_and_save_funding_rate_data(
                    "BTC/USDT:USDT", limit=100
                )

                assert result["success"] is True
                assert result["fetched_count"] == 2
                assert result["saved_count"] == 2

    async def test_fetch_and_save_with_custom_repository(
        self, service, mock_repository, mock_config
    ):
        """カスタムリポジトリでのデータ保存を確認"""
        mock_data = [{"timestamp": 1609459200000, "fundingRate": 0.0001}]

        mock_config.data_converter_class.ccxt_to_db_format = MagicMock(
            return_value=[{"id": 1}]
        )
        mock_repository.insert_funding_rate_data = MagicMock(return_value=1)

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value=mock_data)
            mock_loop.return_value.run_in_executor = mock_executor

            service.exchange.fetch_funding_rate_history.return_value = mock_data

            result = await service.fetch_and_save_funding_rate_data(
                "BTC/USDT:USDT", limit=100, repository=mock_repository
            )

            assert result["success"] is True
            assert result["saved_count"] == 1

    async def test_fetch_and_save_with_fetch_all(self, service, mock_config):
        """fetch_all=Trueでの全期間データ取得を確認"""
        mock_data = [
            {"timestamp": 1609459200000, "fundingRate": 0.0001},
            {"timestamp": 1609459300000, "fundingRate": 0.0002},
        ]

        mock_config.data_converter_class.ccxt_to_db_format = MagicMock(
            return_value=[{"id": 1}, {"id": 2}]
        )

        with patch.object(service, "_get_latest_timestamp_from_db", return_value=None):
            with patch.object(service, "_fetch_paginated_data", return_value=mock_data):
                with patch(
                    "app.services.data_collection.bybit.bybit_service.get_db"
                ) as mock_get_db:
                    mock_db = MagicMock()
                    mock_get_db.return_value = iter([mock_db])
                    mock_repository_instance = MagicMock()
                    mock_repository_instance.insert_funding_rate_data = MagicMock(
                        return_value=2
                    )
                    mock_config.repository_class.return_value = mock_repository_instance

                    result = await service.fetch_and_save_funding_rate_data(
                        "BTC/USDT:USDT", fetch_all=True
                    )

                    assert result["success"] is True
                    assert result["saved_count"] == 2


@pytest.mark.asyncio
class TestDatabaseSave:
    """データベース保存テスト"""

    async def test_save_funding_rate_to_database(
        self, service, mock_repository, mock_config
    ):
        """ファンディングレートデータのDB保存を確認"""
        funding_history = [
            {"timestamp": 1609459200000, "fundingRate": 0.0001},
            {"timestamp": 1609459300000, "fundingRate": 0.0002},
        ]

        mock_config.data_converter_class.ccxt_to_db_format = MagicMock(
            return_value=[{"id": 1}, {"id": 2}]
        )
        mock_repository.insert_funding_rate_data = MagicMock(return_value=2)

        result = await service._save_funding_rate_to_database(
            funding_history, "BTC/USDT:USDT", mock_repository
        )

        assert result == 2
        mock_config.data_converter_class.ccxt_to_db_format.assert_called_once_with(
            funding_history, "BTC/USDT:USDT"
        )
        mock_repository.insert_funding_rate_data.assert_called_once()




