"""
Bybit市場データサービスのテスト

BybitMarketDataServiceクラスの全機能をテストします:
- サービス初期化
- OHLCVデータ取得
- データ検証
- 時間軸検証
- データベース保存
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.data_collection.bybit.market_data_service import (
    BybitMarketDataService,
)


@pytest.fixture
def mock_exchange():
    """モックCCXT取引所"""
    exchange = MagicMock()
    exchange.fetch_ohlcv = AsyncMock()
    return exchange


@pytest.fixture
def service(mock_exchange):
    """サービスインスタンス"""
    with patch("app.services.data_collection.bybit.bybit_service.ccxt.bybit") as mock:
        mock.return_value = mock_exchange

        return BybitMarketDataService()


@pytest.fixture
def mock_repository():
    """モックOHLCVリポジトリ"""
    repo = MagicMock()
    repo.insert_ohlcv_data = MagicMock(return_value=10)
    return repo


@pytest.mark.asyncio
class TestServiceInitialization:
    """サービス初期化テスト"""

    async def test_service_initialization(self, mock_exchange):
        """サービスが正しく初期化されることを確認"""
        with patch(
            "app.services.data_collection.bybit.bybit_service.ccxt.bybit"
        ) as mock:
            mock.return_value = mock_exchange

            service = BybitMarketDataService()

            assert service.exchange is not None
            mock.assert_called_once()

    async def test_service_inherits_from_bybit_service(self, service):
        """BybitServiceを継承していることを確認"""
        from app.services.data_collection.bybit.bybit_service import BybitService

        assert isinstance(service, BybitService)


@pytest.mark.asyncio
class TestFetchOHLCVData:
    """OHLCVデータ取得テスト"""

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    @patch(
        "app.services.data_collection.bybit.market_data_service.unified_config.market.supported_timeframes",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    async def test_fetch_ohlcv_data_success(self, service, mock_exchange):
        """OHLCVデータが正常に取得できることを確認"""
        expected_data = [
            [1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5],
            [1609462800000, 29200.0, 29800.0, 29000.0, 29500.0, 120.3],
        ]

        mock_exchange.fetch_ohlcv.return_value = expected_data

        result = await service.fetch_ohlcv_data(
            symbol="BTC/USD:BTC", timeframe="1h", limit=100
        )

        assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    @patch(
        "app.services.data_collection.bybit.market_data_service.unified_config.market.supported_timeframes",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    async def test_fetch_ohlcv_data_with_since(self, service, mock_exchange):
        """sinceパラメータ付きでOHLCVデータが取得できることを確認"""
        since_timestamp = 1609459200000
        expected_data = [[1609462800000, 29200.0, 29800.0, 29000.0, 29500.0, 120.3]]

        mock_exchange.fetch_ohlcv.return_value = expected_data

        result = await service.fetch_ohlcv_data(
            symbol="BTC/USD:BTC",
            timeframe="1h",
            limit=100,
            since=since_timestamp,
        )

        assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    @patch(
        "app.services.data_collection.bybit.market_data_service.unified_config.market.supported_timeframes",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    async def test_fetch_ohlcv_data_with_params(self, service, mock_exchange):
        """追加パラメータ付きでOHLCVデータが取得できることを確認"""
        expected_data = [[1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5]]
        params = {"type": "linear"}

        mock_exchange.fetch_ohlcv.return_value = expected_data

        result = await service.fetch_ohlcv_data(
            symbol="BTC/USD:BTC", timeframe="1h", limit=100, params=params
        )

        assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.market_data_service.unified_config.market.supported_timeframes",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    async def test_fetch_ohlcv_data_invalid_symbol(self, service):
        """無効なシンボルでValueErrorが発生することを確認"""
        with pytest.raises(ValueError):
            await service.fetch_ohlcv_data(symbol="", timeframe="1h", limit=100)

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    @patch(
        "app.services.data_collection.bybit.market_data_service.unified_config.market.supported_timeframes",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    async def test_fetch_ohlcv_data_invalid_limit(self, service):
        """無効なlimitでValueErrorが発生することを確認"""
        with pytest.raises(ValueError):
            await service.fetch_ohlcv_data(
                symbol="BTC/USD:BTC", timeframe="1h", limit=0
            )

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    @patch(
        "app.services.data_collection.bybit.market_data_service.unified_config.market.supported_timeframes",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    async def test_fetch_ohlcv_data_invalid_timeframe(self, service):
        """無効な時間軸でValueErrorが発生することを確認"""
        with pytest.raises(ValueError, match="無効な時間軸"):
            await service.fetch_ohlcv_data(
                symbol="BTC/USD:BTC", timeframe="invalid", limit=100
            )


@pytest.mark.asyncio
class TestTimeframeValidation:
    """時間軸検証テスト"""

    @patch(
        "app.services.data_collection.bybit.market_data_service.unified_config.market.supported_timeframes",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    async def test_validate_timeframe_valid(self, service):
        """有効な時間軸が検証を通過することを確認"""
        service._validate_timeframe("1h")
        service._validate_timeframe("1d")
        service._validate_timeframe("5m")

    @patch(
        "app.services.data_collection.bybit.market_data_service.unified_config.market.supported_timeframes",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    async def test_validate_timeframe_invalid(self, service):
        """無効な時間軸がValueErrorを発生させることを確認"""
        with pytest.raises(ValueError, match="無効な時間軸"):
            service._validate_timeframe("2h")

    @patch(
        "app.services.data_collection.bybit.market_data_service.unified_config.market.supported_timeframes",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    async def test_validate_timeframe_unsupported(self, service):
        """サポート外の時間軸がValueErrorを発生させることを確認"""
        with pytest.raises(ValueError, match="サポート対象"):
            service._validate_timeframe("3m")


@pytest.mark.asyncio
class TestOHLCVDataValidation:
    """OHLCVデータ検証テスト"""

    async def test_validate_ohlcv_data_valid(self, service):
        """有効なOHLCVデータが検証を通過することを確認"""
        valid_data = [
            [1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5],
            [1609462800000, 29200.0, 29800.0, 29000.0, 29500.0, 120.3],
        ]
        service._validate_ohlcv_data(valid_data)

    async def test_validate_ohlcv_data_empty(self, service):
        """空のOHLCVデータが警告のみで通過することを確認"""
        service._validate_ohlcv_data([])

    async def test_validate_ohlcv_data_not_list(self, service):
        """リストでないデータがValueErrorを発生させることを確認"""
        with pytest.raises(ValueError, match="OHLCVデータはリストである必要があります"):
            service._validate_ohlcv_data("not a list")

    async def test_validate_ohlcv_data_invalid_candle_length(self, service):
        """無効な長さのローソク足データがValueErrorを発生させることを確認"""
        invalid_data = [[1609459200000, 29000.0, 29500.0, 28500.0, 29200.0]]
        with pytest.raises(ValueError, match="ローソク足データ.*の形式が無効です"):
            service._validate_ohlcv_data(invalid_data)

    async def test_validate_ohlcv_data_non_numeric(self, service):
        """非数値データがValueErrorを発生させることを確認"""
        invalid_data = [[1609459200000, "29000.0", 29500.0, 28500.0, 29200.0, 100.5]]
        with pytest.raises(ValueError, match="非数値が含まれています"):
            service._validate_ohlcv_data(invalid_data)

    async def test_validate_ohlcv_data_invalid_high_low(self, service):
        """高値<安値のデータがValueErrorを発生させることを確認"""
        invalid_data = [[1609459200000, 29000.0, 28000.0, 29500.0, 29200.0, 100.5]]
        with pytest.raises(ValueError, match="価格関係が無効です"):
            service._validate_ohlcv_data(invalid_data)

    async def test_validate_ohlcv_data_invalid_price_relationship(self, service):
        """価格関係が無効なデータがValueErrorを発生させることを確認"""
        invalid_data = [[1609459200000, 29000.0, 28500.0, 28500.0, 29200.0, 100.5]]
        with pytest.raises(ValueError, match="価格関係が無効です"):
            service._validate_ohlcv_data(invalid_data)

    async def test_validate_ohlcv_data_negative_price(self, service):
        """負の価格がValueErrorを発生させることを確認"""
        invalid_data = [[1609459200000, -29000.0, 29500.0, 28500.0, 29200.0, 100.5]]
        with pytest.raises(ValueError, match="価格関係が無効です"):
            service._validate_ohlcv_data(invalid_data)

    async def test_validate_ohlcv_data_negative_volume(self, service):
        """負の出来高がValueErrorを発生させることを確認"""
        invalid_data = [[1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, -100.5]]
        with pytest.raises(ValueError, match="負の出来高が含まれています"):
            service._validate_ohlcv_data(invalid_data)


@pytest.mark.asyncio
class TestDatabaseSave:
    """データベース保存テスト"""

    async def test_save_ohlcv_to_database(self, service, mock_repository):
        """OHLCVデータのDB保存が正常に行われることを確認"""
        ohlcv_data = [
            [1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5],
            [1609462800000, 29200.0, 29800.0, 29000.0, 29500.0, 120.3],
        ]

        with patch(
            "app.services.data_collection.bybit.market_data_service.OHLCVDataConverter"
        ) as mock_converter:
            mock_converter.ccxt_to_db_format = MagicMock(
                return_value=[{"id": 1}, {"id": 2}]
            )

            result = await service._save_ohlcv_to_database(
                ohlcv_data, "BTC/USD:BTC", "1h", mock_repository
            )

            assert result == 10
            mock_converter.ccxt_to_db_format.assert_called_once_with(
                ohlcv_data, "BTC/USD:BTC", "1h"
            )
            mock_repository.insert_ohlcv_data.assert_called_once()

    async def test_save_ohlcv_to_database_empty_data(self, service, mock_repository):
        """空のOHLCVデータでもDB保存が実行されることを確認"""
        ohlcv_data = []

        with patch(
            "app.services.data_collection.bybit.market_data_service.OHLCVDataConverter"
        ) as mock_converter:
            mock_converter.ccxt_to_db_format = MagicMock(return_value=[])
            mock_repository.insert_ohlcv_data = MagicMock(return_value=0)

            result = await service._save_ohlcv_to_database(
                ohlcv_data, "BTC/USD:BTC", "1h", mock_repository
            )

            assert result == 0

    async def test_save_ohlcv_to_database_multiple_timeframes(
        self, service, mock_repository
    ):
        """複数時間軸のデータが正しく保存されることを確認"""
        ohlcv_data = [[1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5]]

        with patch(
            "app.services.data_collection.bybit.market_data_service.OHLCVDataConverter"
        ) as mock_converter:
            mock_converter.ccxt_to_db_format = MagicMock(return_value=[{"id": 1}])

            # 1時間足
            result_1h = await service._save_ohlcv_to_database(
                ohlcv_data, "BTC/USD:BTC", "1h", mock_repository
            )

            # 4時間足
            result_4h = await service._save_ohlcv_to_database(
                ohlcv_data, "BTC/USD:BTC", "4h", mock_repository
            )

            assert result_1h == 10
            assert result_4h == 10


@pytest.mark.asyncio
class TestEdgeCases:
    """エッジケーステスト"""

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    @patch(
        "app.services.data_collection.bybit.market_data_service.unified_config.market.supported_timeframes",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    async def test_fetch_ohlcv_data_max_limit(self, service, mock_exchange):
        """最大limitでのOHLCVデータ取得を確認"""
        expected_data = [[1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 100.5]]

        mock_exchange.fetch_ohlcv.return_value = expected_data

        result = await service.fetch_ohlcv_data(
            symbol="BTC/USD:BTC", timeframe="1h", limit=1000
        )

        assert result == expected_data

    @patch(
        "app.services.data_collection.bybit.bybit_service.unified_config.data_collection.max_limit",
        1000,
    )
    @patch(
        "app.services.data_collection.bybit.market_data_service.unified_config.market.supported_timeframes",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    async def test_fetch_ohlcv_data_zero_volume(self, service, mock_exchange):
        """ゼロ出来高のOHLCVデータが取得できることを確認"""
        expected_data = [[1609459200000, 29000.0, 29500.0, 28500.0, 29200.0, 0.0]]

        mock_exchange.fetch_ohlcv.return_value = expected_data

        result = await service.fetch_ohlcv_data(
            symbol="BTC/USD:BTC", timeframe="1h", limit=100
        )

        assert result == expected_data
        service._validate_ohlcv_data(result)

    async def test_validate_ohlcv_data_high_equals_low(self, service):
        """高値=安値のデータが検証を通過することを確認"""
        valid_data = [[1609459200000, 29000.0, 29000.0, 29000.0, 29000.0, 100.5]]
        service._validate_ohlcv_data(valid_data)

    async def test_validate_ohlcv_data_large_numbers(self, service):
        """大きな数値のデータが検証を通過することを確認"""
        valid_data = [
            [
                1609459200000,
                1000000.0,
                1100000.0,
                900000.0,
                1050000.0,
                99999999.99,
            ]
        ]
        service._validate_ohlcv_data(valid_data)
