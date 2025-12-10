"""
Bybitオープンインタレストサービスのテスト

BybitOpenInterestServiceクラスの全機能をテストします:
- サービス初期化
- オープンインタレスト履歴取得
- 差分更新
- データベース保存
- インターバル設定
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.data_collection.bybit.open_interest_service import (
    BybitOpenInterestService,
)


@pytest.fixture
def mock_exchange():
    """モックCCXT取引所"""
    exchange = MagicMock()
    exchange.fetch_open_interest_history = AsyncMock()
    return exchange


@pytest.fixture
def mock_config():
    """モック設定"""
    config = MagicMock()
    config.repository_class = MagicMock()
    config.get_timestamp_method_name = "get_latest_open_interest_timestamp"
    config.data_converter_class = MagicMock()
    config.converter_method_name = "ccxt_to_db_format"
    config.fetch_history_method_name = "fetch_open_interest_history"
    config.fetch_current_method_name = "fetch_open_interest"
    config.pagination_strategy = "time_range"
    config.default_limit = 100
    config.page_limit = 200
    config.max_pages = 500
    config.insert_method_name = "insert_open_interest_data"
    config.log_prefix = "OI"
    return config


@pytest.fixture
def service(mock_exchange, mock_config):
    """サービスインスタンス"""
    with patch(
        "app.services.data_collection.bybit.bybit_service.ccxt.bybit"
    ) as mock_ccxt:
        mock_ccxt.return_value = mock_exchange
        with patch(
            "app.services.data_collection.bybit.open_interest_service.get_open_interest_config"
        ) as mock_get_config:
            mock_get_config.return_value = mock_config
            return BybitOpenInterestService()


@pytest.fixture
def mock_repository():
    """モックリポジトリ"""
    repo = MagicMock()
    repo.get_latest_open_interest_timestamp = MagicMock()
    repo.insert_open_interest_data = MagicMock(return_value=10)
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
                "app.services.data_collection.bybit.open_interest_service.get_open_interest_config"
            ) as mock_get_config:
                mock_get_config.return_value = mock_config

                service = BybitOpenInterestService()

                assert service.exchange is not None
                assert service.config == mock_config
                mock_get_config.assert_called_once()

    async def test_config_loaded(self, service, mock_config):
        """設定が正しく読み込まれることを確認"""
        assert service.config == mock_config
        assert service.config.log_prefix == "OI"
        assert service.config.pagination_strategy == "time_range"


@pytest.mark.asyncio
class TestFetchOpenInterestHistory:
    """オープンインタレスト履歴取得テスト"""

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_fetch_open_interest_history_success(self, service, mock_exchange):
        """オープンインタレスト履歴が正常に取得できることを確認"""
        expected_data = [
            {"timestamp": 1609459200000, "openInterest": 100000.0},
            {"timestamp": 1609462800000, "openInterest": 105000.0},
        ]

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value=expected_data)
            mock_loop.return_value.run_in_executor = mock_executor

            mock_exchange.fetch_open_interest_history.return_value = expected_data

            result = await service.fetch_open_interest_history(
                "BTC/USDT", limit=100, interval="1h"
            )

            assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_fetch_open_interest_history_with_since(self, service, mock_exchange):
        """sinceパラメータ付きでオープンインタレスト履歴が取得できることを確認"""
        since_timestamp = 1609459200000
        expected_data = [{"timestamp": 1609462800000, "openInterest": 105000.0}]

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value=expected_data)
            mock_loop.return_value.run_in_executor = mock_executor

            mock_exchange.fetch_open_interest_history.return_value = expected_data

            result = await service.fetch_open_interest_history(
                "BTC/USDT", limit=100, since=since_timestamp, interval="1h"
            )

            assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_fetch_open_interest_history_different_intervals(
        self, service, mock_exchange
    ):
        """異なるインターバルでの取得を確認"""
        expected_data = [{"timestamp": 1609459200000, "openInterest": 100000.0}]

        intervals = ["5min", "15min", "30min", "1h", "4h", "1d"]

        for interval in intervals:
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = AsyncMock(return_value=expected_data)
                mock_loop.return_value.run_in_executor = mock_executor

                mock_exchange.fetch_open_interest_history.return_value = expected_data

                result = await service.fetch_open_interest_history(
                    "BTC/USDT", limit=100, interval=interval
                )

                assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_fetch_open_interest_history_invalid_limit(self, service):
        """無効なlimitでValueErrorが発生することを確認"""
        with pytest.raises(ValueError):
            await service.fetch_open_interest_history("BTC/USDT", limit=0)

    async def test_fetch_open_interest_history_invalid_symbol(self, service):
        """無効なシンボルでValueErrorが発生することを確認"""
        with pytest.raises(ValueError):
            await service.fetch_open_interest_history("", limit=100)


@pytest.mark.asyncio
class TestFetchIncrementalData:
    """差分データ取得テスト"""

    async def test_fetch_incremental_open_interest_data_no_existing(
        self, service, mock_repository, mock_config
    ):
        """既存データなしでの差分更新を確認"""
        mock_data = [
            {"timestamp": 1609459200000, "openInterest": 100000.0},
            {"timestamp": 1609462800000, "openInterest": 105000.0},
        ]
        service.exchange.fetch_open_interest_history.return_value = mock_data

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
                    mock_repository_instance.insert_open_interest_data = MagicMock(
                        return_value=2
                    )
                    mock_config.repository_class.return_value = mock_repository_instance

                    result = await service.fetch_incremental_open_interest_data(
                        "BTC/USDT"
                    )

                    assert result["success"] is True
                    assert result["saved_count"] == 2

    async def test_fetch_incremental_open_interest_data_with_existing(
        self, service, mock_repository, mock_config
    ):
        """既存データありでの差分更新を確認"""
        latest_timestamp = 1609459200000
        mock_data = [{"timestamp": 1609462800000, "openInterest": 105000.0}]
        service.exchange.fetch_open_interest_history.return_value = mock_data

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
                    mock_repository_instance.insert_open_interest_data = MagicMock(
                        return_value=1
                    )
                    mock_config.repository_class.return_value = mock_repository_instance

                    result = await service.fetch_incremental_open_interest_data(
                        "BTC/USDT"
                    )

                    assert result["success"] is True
                    assert result["saved_count"] == 1
                    assert result["latest_timestamp"] == latest_timestamp

    async def test_fetch_incremental_with_custom_interval(
        self, service, mock_repository, mock_config
    ):
        """カスタムインターバルでの差分更新を確認"""
        mock_data = [{"timestamp": 1609459200000, "openInterest": 100000.0}]
        service.exchange.fetch_open_interest_history.return_value = mock_data

        mock_config.data_converter_class.ccxt_to_db_format = MagicMock(
            return_value=[{"id": 1}]
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
                    mock_repository_instance.insert_open_interest_data = MagicMock(
                        return_value=1
                    )
                    mock_config.repository_class.return_value = mock_repository_instance

                    result = await service.fetch_incremental_open_interest_data(
                        "BTC/USDT", interval="4h"
                    )

                    assert result["success"] is True
                    assert result["saved_count"] == 1

    async def test_fetch_incremental_with_custom_repository(
        self, service, mock_repository, mock_config
    ):
        """カスタムリポジトリでの差分更新を確認"""
        mock_data = [{"timestamp": 1609459200000, "openInterest": 100000.0}]

        mock_config.data_converter_class.ccxt_to_db_format = MagicMock(
            return_value=[{"id": 1}]
        )
        mock_repository.insert_open_interest_data = MagicMock(return_value=1)

        with patch.object(service, "_get_latest_timestamp_from_db", return_value=None):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = AsyncMock(return_value=mock_data)
                mock_loop.return_value.run_in_executor = mock_executor

                service.exchange.fetch_open_interest_history.return_value = mock_data

                result = await service.fetch_incremental_open_interest_data(
                    "BTC/USDT", mock_repository
                )

                assert result["success"] is True
                assert result["saved_count"] == 1


@pytest.mark.asyncio
class TestFetchAndSaveData:
    """データ取得・保存テスト"""

    async def test_fetch_and_save_open_interest_data_success(
        self, service, mock_exchange, mock_config
    ):
        """オープンインタレストデータの取得・保存が成功することを確認"""
        mock_data = [
            {"timestamp": 1609459200000, "openInterest": 100000.0},
            {"timestamp": 1609462800000, "openInterest": 105000.0},
        ]
        service.exchange.fetch_open_interest_history.return_value = mock_data

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
                mock_repository_instance.insert_open_interest_data = MagicMock(
                    return_value=2
                )
                mock_config.repository_class.return_value = mock_repository_instance

                result = await service.fetch_and_save_open_interest_data(
                    "BTC/USDT", limit=100
                )

                assert result["success"] is True
                assert result["fetched_count"] == 2
                assert result["saved_count"] == 2

    async def test_fetch_and_save_with_custom_repository(
        self, service, mock_repository, mock_config
    ):
        """カスタムリポジトリでのデータ保存を確認"""
        mock_data = [{"timestamp": 1609459200000, "openInterest": 100000.0}]
        service.exchange.fetch_open_interest_history.return_value = mock_data

        mock_config.data_converter_class.ccxt_to_db_format = MagicMock(
            return_value=[{"id": 1}]
        )
        mock_repository.insert_open_interest_data = MagicMock(return_value=1)

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value=mock_data)
            mock_loop.return_value.run_in_executor = mock_executor

            result = await service.fetch_and_save_open_interest_data(
                "BTC/USDT", limit=100, repository=mock_repository
            )

            assert result["success"] is True
            assert result["saved_count"] == 1

    async def test_fetch_and_save_with_fetch_all(self, service, mock_config):
        """fetch_all=Trueでの全期間データ取得を確認"""
        mock_data = [
            {"timestamp": 1609459200000, "openInterest": 100000.0},
            {"timestamp": 1609462800000, "openInterest": 105000.0},
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
                    mock_repository_instance.insert_open_interest_data = MagicMock(
                        return_value=2
                    )
                    mock_config.repository_class.return_value = mock_repository_instance

                    result = await service.fetch_and_save_open_interest_data(
                        "BTC/USDT", fetch_all=True
                    )

                    assert result["success"] is True
                    assert result["saved_count"] == 2

    async def test_fetch_and_save_with_custom_interval(
        self, service, mock_exchange, mock_config
    ):
        """カスタムインターバルでのデータ保存を確認"""
        mock_data = [{"timestamp": 1609459200000, "openInterest": 100000.0}]
        service.exchange.fetch_open_interest_history.return_value = mock_data

        mock_config.data_converter_class.ccxt_to_db_format = MagicMock(
            return_value=[{"id": 1}]
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
                mock_repository_instance.insert_open_interest_data = MagicMock(
                    return_value=1
                )
                mock_config.repository_class.return_value = mock_repository_instance

                result = await service.fetch_and_save_open_interest_data(
                    "BTC/USDT", limit=100, interval="4h"
                )

                assert result["success"] is True
                assert result["saved_count"] == 1


@pytest.mark.asyncio
class TestIntervalHandling:
    """インターバル処理テスト"""

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_default_interval(self, service, mock_exchange):
        """デフォルトインターバル（1h）が使用されることを確認"""
        expected_data = [{"timestamp": 1609459200000, "openInterest": 100000.0}]

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value=expected_data)
            mock_loop.return_value.run_in_executor = mock_executor

            mock_exchange.fetch_open_interest_history.return_value = expected_data

            result = await service.fetch_open_interest_history("BTC/USDT", limit=100)

            assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_multiple_intervals(self, service, mock_exchange):
        """複数の異なるインターバルが正しく処理されることを確認"""
        expected_data = [{"timestamp": 1609459200000, "openInterest": 100000.0}]

        test_intervals = ["5min", "15min", "1h", "4h", "1d"]

        for interval in test_intervals:
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = AsyncMock(return_value=expected_data)
                mock_loop.return_value.run_in_executor = mock_executor

                mock_exchange.fetch_open_interest_history.return_value = expected_data

                result = await service.fetch_open_interest_history(
                    "BTC/USDT", limit=100, interval=interval
                )

                assert result == expected_data


@pytest.mark.asyncio
class TestEdgeCases:
    """エッジケーステスト"""

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_fetch_with_max_limit(self, service, mock_exchange):
        """最大limitでのオープンインタレスト取得を確認"""
        expected_data = [{"timestamp": 1609459200000, "openInterest": 100000.0}]

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value=expected_data)
            mock_loop.return_value.run_in_executor = mock_executor

            mock_exchange.fetch_open_interest_history.return_value = expected_data

            result = await service.fetch_open_interest_history(
                "BTC/USDT", limit=1000, interval="1h"
            )

            assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_fetch_with_zero_open_interest(self, service, mock_exchange):
        """ゼロ建玉のデータが取得できることを確認"""
        expected_data = [{"timestamp": 1609459200000, "openInterest": 0.0}]

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value=expected_data)
            mock_loop.return_value.run_in_executor = mock_executor

            mock_exchange.fetch_open_interest_history.return_value = expected_data

            result = await service.fetch_open_interest_history(
                "BTC/USDT", limit=100, interval="1h"
            )

            assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    async def test_fetch_with_large_open_interest(self, service, mock_exchange):
        """大きな建玉値のデータが取得できることを確認"""
        expected_data = [{"timestamp": 1609459200000, "openInterest": 999999999.99}]

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(return_value=expected_data)
            mock_loop.return_value.run_in_executor = mock_executor

            mock_exchange.fetch_open_interest_history.return_value = expected_data

            result = await service.fetch_open_interest_history(
                "BTC/USDT", limit=100, interval="1h"
            )

            assert result == expected_data
