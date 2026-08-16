"""
履歴データ収集サービスのテスト

HistoricalDataServiceクラスの全機能をテストします:
- サービス初期化
- 履歴データ収集
- ページネーション処理
- 一括差分データ収集
- エラーハンドリング
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import ccxt
import pytest

from app.services.data_collection.historical.historical_data_service import (
    HistoricalDataService,
)


@pytest.fixture
def mock_market_service():
    """モック市場データサービス"""
    service = MagicMock()
    service.fetch_ohlcv_data = AsyncMock()
    service._save_ohlcv_to_database = AsyncMock(return_value=10)
    return service


@pytest.fixture
def service(mock_market_service):
    """サービスインスタンス"""
    return HistoricalDataService(mock_market_service)


@pytest.fixture
def mock_repository():
    """モックリポジトリ"""
    repo = MagicMock()
    repo.get_latest_timestamp = MagicMock()
    repo.insert_ohlcv_data = MagicMock(return_value=10)
    return repo


@pytest.mark.asyncio
class TestServiceInitialization:
    """サービス初期化テスト"""

    async def test_service_initialization_with_market_service(
        self, mock_market_service
    ):
        """市場データサービス付きで初期化されることを確認"""
        service = HistoricalDataService(mock_market_service)
        assert service.market_service == mock_market_service
        assert service.request_delay == 0.2

    async def test_service_initialization_without_market_service(self):
        """市場データサービスなしで初期化されることを確認"""
        with patch(
            "app.services.data_collection.bybit.market_data_service.BybitMarketDataService"
        ):
            service = HistoricalDataService()
            assert service.market_service is not None


@pytest.mark.asyncio
class TestCollectHistoricalData:
    """履歴データ収集テスト"""

    async def test_collect_historical_data_success(
        self, service, mock_market_service, mock_repository
    ):
        """履歴データ収集が成功することを確認"""
        mock_data = [
            [1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5],
            [1609462800000, 29200.0, 29800.0, 29000.0, 29500.0, 120.3],
        ]
        mock_market_service.fetch_ohlcv_data.return_value = mock_data
        mock_repository.get_latest_timestamp.return_value = None

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await service.collect_historical_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                repository=mock_repository,
            )

            assert result == 10
            mock_market_service.fetch_ohlcv_data.assert_called()

    async def test_collect_historical_data_with_pagination(
        self, service, mock_market_service, mock_repository
    ):
        """ページネーション処理が正しく行われることを確認"""
        # 最初のページ: 完全なデータ
        first_page = [
            [i, 29000.0, 29500.0, 28500.0, 29200.0, 100.5] for i in range(1000)
        ]
        # 2ページ目: データなし（終了）
        mock_market_service.fetch_ohlcv_data.side_effect = [first_page, []]
        mock_repository.get_latest_timestamp.return_value = None

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await service.collect_historical_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                repository=mock_repository,
            )

            assert result == 10

    async def test_collect_historical_data_no_repository(self, service):
        """リポジトリなしでValueErrorが発生することを確認"""
        with pytest.raises(ValueError, match="リポジトリが必要です"):
            await service.collect_historical_data()

    async def test_collect_historical_data_network_error(
        self, service, mock_market_service, mock_repository
    ):
        """ネットワークエラーが適切に処理されることを確認"""
        mock_market_service.fetch_ohlcv_data.side_effect = ccxt.NetworkError(
            "Connection failed"
        )
        mock_repository.get_latest_timestamp.return_value = None

        with pytest.raises(ccxt.NetworkError):
            await service.collect_historical_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                repository=mock_repository,
            )

    async def test_collect_historical_data_exchange_error(
        self, service, mock_market_service, mock_repository
    ):
        """取引所エラーが適切に処理されることを確認"""
        mock_market_service.fetch_ohlcv_data.side_effect = ccxt.ExchangeError(
            "Exchange error"
        )
        mock_repository.get_latest_timestamp.return_value = None

        with pytest.raises(ccxt.ExchangeError):
            await service.collect_historical_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                repository=mock_repository,
            )


@pytest.mark.asyncio
class TestCollectBulkIncrementalData:
    """一括差分データ収集テスト"""

    async def test_collect_bulk_incremental_data_success(
        self, service, mock_market_service, mock_repository
    ):
        """一括差分データ収集が成功することを確認"""
        mock_data = [[1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5]]
        mock_market_service.fetch_ohlcv_data.return_value = mock_data
        mock_repository.get_latest_timestamp.return_value = None

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await service.collect_bulk_incremental_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                ohlcv_repository=mock_repository,
            )

            assert result["success"] is True
            assert "data" in result
            assert "ohlcv" in result["data"]

    async def test_collect_bulk_incremental_data_all_timeframes(
        self, service, mock_market_service, mock_repository
    ):
        """全時間足でデータ収集が行われることを確認"""
        mock_data = [[1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5]]
        mock_market_service.fetch_ohlcv_data.return_value = mock_data
        mock_repository.get_latest_timestamp.return_value = None

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await service.collect_bulk_incremental_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                ohlcv_repository=mock_repository,
            )

            assert "ohlcv" in result["data"]
            assert "timeframe_results" in result["data"]["ohlcv"]
            # 7つの時間足（1m, 5m, 15m, 30m, 1h, 4h, 1d）をテスト
            assert len(result["data"]["ohlcv"]["timeframe_results"]) == 7

    async def test_collect_bulk_incremental_data_with_funding_rate(
        self, service, mock_market_service, mock_repository
    ):
        """ファンディングレートを含む一括収集を確認"""
        mock_data = [[1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5]]
        mock_market_service.fetch_ohlcv_data.return_value = mock_data
        mock_repository.get_latest_timestamp.return_value = None

        mock_fr_repo = MagicMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch(
                "app.services.data_collection.bybit.funding_rate_service.BybitFundingRateService"
            ) as mock_fr_service_class:
                mock_fr_service = MagicMock()
                mock_fr_service.fetch_incremental_funding_rate_data = AsyncMock(
                    return_value={
                        "symbol": "BTC/USDT:USDT",
                        "saved_count": 5,
                        "success": True,
                    }
                )
                mock_fr_service_class.return_value = mock_fr_service

                result = await service.collect_bulk_incremental_data(
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    ohlcv_repository=mock_repository,
                    funding_rate_repository=mock_fr_repo,
                )

                assert "funding_rate" in result["data"]
                assert result["data"]["funding_rate"]["saved_count"] == 5

    async def test_collect_bulk_incremental_data_pages_through_large_gap(
        self, service, mock_market_service, mock_repository
    ):
        """1000件超の差分がある場合にページングで全件取得することを確認"""
        base_ts = 1_700_000_000_000
        page1 = [
            [base_ts + i * 60000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5]
            for i in range(1000)
        ]
        page2 = [
            [
                base_ts + 59_940_001 + j * 60000,
                29000.0,
                29500.0,
                28500.0,
                29200.0,
                100.5,
            ]
            for j in range(500)
        ]
        mock_market_service.fetch_ohlcv_data.side_effect = [page1, page2]
        mock_repository.get_latest_timestamp.return_value = datetime(2023, 11, 14)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch("app.config.unified_config.unified_config") as mock_config:
                mock_config.market.supported_timeframes = ["1h"]
                result = await service.collect_bulk_incremental_data(
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    ohlcv_repository=mock_repository,
                )

        # 2ページ取得し、2ページ目は1ページ目の最終タイムスタンプ+1から始まる
        assert mock_market_service.fetch_ohlcv_data.call_count == 2
        second_call = mock_market_service.fetch_ohlcv_data.call_args_list[1]
        assert second_call.kwargs["since"] == base_ts + 59_940_001
        # 保存はページごとに呼ばれ、件数は合算される
        assert mock_market_service._save_ohlcv_to_database.call_count == 2
        tf_result = result["data"]["ohlcv"]["timeframe_results"]["1h"]
        assert tf_result["saved_count"] == 20

    async def test_collect_bulk_incremental_data_stops_when_page_not_full(
        self, service, mock_market_service, mock_repository
    ):
        """ページが1000件未満なら最新到達とみなし追加取得しないことを確認"""
        short_page = [
            [1_700_000_000_000 + i * 60000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5]
            for i in range(300)
        ]
        mock_market_service.fetch_ohlcv_data.return_value = short_page
        mock_repository.get_latest_timestamp.return_value = datetime(2023, 11, 14)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch("app.config.unified_config.unified_config") as mock_config:
                mock_config.market.supported_timeframes = ["1h"]
                await service.collect_bulk_incremental_data(
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    ohlcv_repository=mock_repository,
                )

        assert mock_market_service.fetch_ohlcv_data.call_count == 1

    async def test_collect_bulk_incremental_data_stops_on_stale_page(
        self, service, mock_market_service, mock_repository
    ):
        """タイムスタンプが進まないページでは無限ループせず終了することを確認"""
        stale_page = [
            [1_700_000_000_000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5]
            for _ in range(1000)
        ]
        mock_market_service.fetch_ohlcv_data.return_value = stale_page
        mock_repository.get_latest_timestamp.return_value = datetime(2023, 11, 14)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch("app.config.unified_config.unified_config") as mock_config:
                mock_config.market.supported_timeframes = ["1h"]
                await service.collect_bulk_incremental_data(
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    ohlcv_repository=mock_repository,
                )

        # 2ページ目でタイムスタンプ不進行を検知して終了する
        assert mock_market_service.fetch_ohlcv_data.call_count == 2


@pytest.mark.asyncio
class TestCollectHistoricalDataWithStartDate:
    """開始日付指定での履歴データ収集テスト"""

    async def test_collect_with_start_date_success(
        self, service, mock_market_service, mock_repository
    ):
        """開始日付指定でのデータ収集が成功することを確認"""
        mock_data = [[1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5]]
        mock_market_service.fetch_ohlcv_data.return_value = mock_data
        mock_repository.get_latest_timestamp.return_value = None

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await service.collect_historical_data_with_start_date(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                repository=mock_repository,
            )

            assert result == 10

    async def test_collect_with_start_date_pagination(
        self, service, mock_market_service, mock_repository
    ):
        """ページネーションが正しく行われることを確認"""
        # 最初: 1000件、次: データなし
        first_page = [
            [i, 29000.0, 29500.0, 28500.0, 29200.0, 100.5] for i in range(1000)
        ]
        mock_market_service.fetch_ohlcv_data.side_effect = [first_page, []]
        mock_repository.get_latest_timestamp.return_value = None

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await service.collect_historical_data_with_start_date(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                repository=mock_repository,
            )

            assert result == 10


@pytest.mark.asyncio
class TestErrorHandling:
    """エラーハンドリングテスト"""

    async def test_handle_bad_symbol_error(
        self, service, mock_market_service, mock_repository
    ):
        """BadSymbolエラーが適切に処理されることを確認"""
        mock_market_service.fetch_ohlcv_data.side_effect = ccxt.BadSymbol(
            "Invalid symbol"
        )
        mock_repository.get_latest_timestamp.return_value = None

        with pytest.raises(ccxt.BadSymbol):
            await service.collect_historical_data(
                symbol="INVALID",
                timeframe="1h",
                repository=mock_repository,
            )

    async def test_handle_generic_exception(
        self, service, mock_market_service, mock_repository
    ):
        """一般的な例外が適切に処理されることを確認"""
        mock_market_service.fetch_ohlcv_data.side_effect = Exception("Unexpected error")
        mock_repository.get_latest_timestamp.return_value = None

        result = await service.collect_historical_data(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            repository=mock_repository,
        )

        assert result == 0


@pytest.mark.asyncio
class TestEdgeCases:
    """エッジケーステスト"""

    async def test_collect_with_existing_data(
        self, service, mock_market_service, mock_repository
    ):
        """既存データありでの収集を確認"""
        mock_data = [[1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5]]
        mock_market_service.fetch_ohlcv_data.return_value = mock_data

        # 既存データのタイムスタンプを設定
        existing_timestamp = datetime.fromtimestamp(1609459200)
        mock_repository.get_latest_timestamp.return_value = existing_timestamp

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await service.collect_historical_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                repository=mock_repository,
            )

            assert result >= 0

    async def test_collect_with_no_new_data(
        self, service, mock_market_service, mock_repository
    ):
        """新規データなしの場合を確認"""
        mock_market_service.fetch_ohlcv_data.return_value = []
        mock_repository.get_latest_timestamp.return_value = None

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await service.collect_historical_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                repository=mock_repository,
            )

            assert result == 0
