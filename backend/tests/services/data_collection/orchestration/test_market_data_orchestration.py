"""
市場データオーケストレーションサービスのテストモジュール

MarketDataOrchestrationServiceの正常系、異常系、エッジケースをテストします。
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.services.data_collection.orchestration.market_data_orchestration_service import (
    MarketDataOrchestrationService,
)


@pytest.fixture
def mock_db_session() -> MagicMock:
    """
    データベースセッションのモック

    Returns:
        MagicMock: モックされたデータベースセッション
    """
    return MagicMock()


@pytest.fixture
def orchestration_service(mock_db_session: MagicMock) -> MarketDataOrchestrationService:
    """
    MarketDataOrchestrationServiceのインスタンス

    Args:
        mock_db_session: DBセッションモック

    Returns:
        MarketDataOrchestrationService: テスト対象のサービス
    """
    return MarketDataOrchestrationService(db_session=mock_db_session)


@pytest.fixture
def sample_ohlcv_data() -> List[Dict[str, Any]]:
    """
    サンプルOHLCVデータ

    Returns:
        List[Dict[str, Any]]: OHLCVデータのリスト
    """
    return [
        {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "timestamp": datetime(2024, 1, 1, 0, 0, 0),
            "open": 40000.0,
            "high": 41000.0,
            "low": 39500.0,
            "close": 40500.0,
            "volume": 1000.0,
        },
        {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "timestamp": datetime(2024, 1, 1, 1, 0, 0),
            "open": 40500.0,
            "high": 41500.0,
            "low": 40000.0,
            "close": 41000.0,
            "volume": 1200.0,
        },
    ]


class TestServiceInitialization:
    """正常系: サービスの初期化テスト"""

    def test_service_creation(
        self, orchestration_service: MarketDataOrchestrationService
    ):
        """
        正常系: サービスが正常に初期化される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        assert orchestration_service is not None
        assert isinstance(orchestration_service, MarketDataOrchestrationService)
        assert orchestration_service.db_session is not None
        assert orchestration_service.repository is not None


class TestGetOHLCVData:
    """正常系: OHLCVデータ取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_success(
        self,
        orchestration_service: MarketDataOrchestrationService,
        sample_ohlcv_data: List[Dict[str, Any]],
    ):
        """
        正常系: OHLCVデータが正常に取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            sample_ohlcv_data: サンプルデータ
        """
        orchestration_service.repository.get_latest_ohlcv_data = MagicMock(
            return_value=sample_ohlcv_data
        )

        with patch(
            "app.services.data_collection.orchestration.market_data_orchestration_service.OHLCVDataConverter"
        ) as mock_converter:
            mock_converter.db_to_api_format.return_value = sample_ohlcv_data

            result = await orchestration_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                limit=100,
            )

            assert result["success"] is True
            assert len(result["data"]["ohlcv_data"]) == 2
            assert result["data"]["symbol"] == "BTC/USDT:USDT"

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_with_date_range(
        self,
        orchestration_service: MarketDataOrchestrationService,
        sample_ohlcv_data: List[Dict[str, Any]],
    ):
        """
        正常系: 日付範囲を指定してOHLCVデータが取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            sample_ohlcv_data: サンプルデータ
        """
        orchestration_service.repository.get_ohlcv_data = MagicMock(
            return_value=sample_ohlcv_data
        )

        with patch(
            "app.services.data_collection.orchestration.market_data_orchestration_service.OHLCVDataConverter"
        ) as mock_converter:
            mock_converter.db_to_api_format.return_value = sample_ohlcv_data

            result = await orchestration_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                limit=100,
                start_date="2024-01-01T00:00:00",
                end_date="2024-01-31T00:00:00",
            )

            assert result["success"] is True
            assert len(result["data"]["ohlcv_data"]) == 2

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_empty_result(
        self,
        orchestration_service: MarketDataOrchestrationService,
    ):
        """
        エッジケース: データが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        orchestration_service.repository.get_latest_ohlcv_data = MagicMock(
            return_value=[]
        )

        with patch(
            "app.services.data_collection.orchestration.market_data_orchestration_service.OHLCVDataConverter"
        ) as mock_converter:
            mock_converter.db_to_api_format.return_value = []

            result = await orchestration_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                limit=100,
            )

            assert result["success"] is True
            assert len(result["data"]["ohlcv_data"]) == 0


class TestDataFiltering:
    """正常系: データフィルタリングのテスト"""

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_with_limit(
        self,
        orchestration_service: MarketDataOrchestrationService,
        sample_ohlcv_data: List[Dict[str, Any]],
    ):
        """
        正常系: limit指定でデータが取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            sample_ohlcv_data: サンプルデータ
        """
        orchestration_service.repository.get_latest_ohlcv_data = MagicMock(
            return_value=sample_ohlcv_data[:1]
        )

        with patch(
            "app.services.data_collection.orchestration.market_data_orchestration_service.OHLCVDataConverter"
        ) as mock_converter:
            mock_converter.db_to_api_format.return_value = sample_ohlcv_data[:1]

            result = await orchestration_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                limit=1,
            )

            assert result["success"] is True
            assert len(result["data"]["ohlcv_data"]) == 1

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_different_timeframes(
        self,
        orchestration_service: MarketDataOrchestrationService,
        sample_ohlcv_data: List[Dict[str, Any]],
    ):
        """
        正常系: 異なる時間軸でデータが取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            sample_ohlcv_data: サンプルデータ
        """
        timeframes = ["15m", "30m", "1h", "4h", "1d"]

        for timeframe in timeframes:
            orchestration_service.repository.get_latest_ohlcv_data = MagicMock(
                return_value=sample_ohlcv_data
            )

            with patch(
                "app.services.data_collection.orchestration.market_data_orchestration_service.OHLCVDataConverter"
            ) as mock_converter:
                mock_converter.db_to_api_format.return_value = sample_ohlcv_data

                result = await orchestration_service.get_ohlcv_data(
                    symbol="BTC/USDT:USDT",
                    timeframe=timeframe,
                    limit=100,
                )

                assert result["success"] is True
                assert result["data"]["timeframe"] == timeframe


class TestErrorHandling:
    """異常系: エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_with_exception(
        self,
        orchestration_service: MarketDataOrchestrationService,
    ):
        """
        異常系: データ取得中に例外が発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        orchestration_service.repository.get_latest_ohlcv_data = MagicMock(
            side_effect=Exception("Database error")
        )

        with pytest.raises(Exception, match="Database error"):
            await orchestration_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                limit=100,
            )

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_invalid_date_format(
        self,
        orchestration_service: MarketDataOrchestrationService,
    ):
        """
        異常系: 無効な日付形式が渡された場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with pytest.raises(Exception):
            await orchestration_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                limit=100,
                start_date="invalid-date",
            )


class TestEdgeCases:
    """境界値テスト"""

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_with_zero_limit(
        self,
        orchestration_service: MarketDataOrchestrationService,
    ):
        """
        境界値: limit=0の場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        orchestration_service.repository.get_latest_ohlcv_data = MagicMock(
            return_value=[]
        )

        with patch(
            "app.services.data_collection.orchestration.market_data_orchestration_service.OHLCVDataConverter"
        ) as mock_converter:
            mock_converter.db_to_api_format.return_value = []

            result = await orchestration_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                limit=0,
            )

            assert result["success"] is True
            assert len(result["data"]["ohlcv_data"]) == 0

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_with_large_limit(
        self,
        orchestration_service: MarketDataOrchestrationService,
        sample_ohlcv_data: List[Dict[str, Any]],
    ):
        """
        境界値: 非常に大きなlimit値

        Args:
            orchestration_service: オーケストレーションサービス
            sample_ohlcv_data: サンプルデータ
        """
        orchestration_service.repository.get_latest_ohlcv_data = MagicMock(
            return_value=sample_ohlcv_data
        )

        with patch(
            "app.services.data_collection.orchestration.market_data_orchestration_service.OHLCVDataConverter"
        ) as mock_converter:
            mock_converter.db_to_api_format.return_value = sample_ohlcv_data

            result = await orchestration_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                limit=999999,
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_same_start_end_date(
        self,
        orchestration_service: MarketDataOrchestrationService,
        sample_ohlcv_data: List[Dict[str, Any]],
    ):
        """
        境界値: 開始日と終了日が同じ場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_ohlcv_data: サンプルデータ
        """
        orchestration_service.repository.get_ohlcv_data = MagicMock(
            return_value=sample_ohlcv_data
        )

        with patch(
            "app.services.data_collection.orchestration.market_data_orchestration_service.OHLCVDataConverter"
        ) as mock_converter:
            mock_converter.db_to_api_format.return_value = sample_ohlcv_data

            same_date = "2024-01-01T00:00:00"
            result = await orchestration_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                limit=100,
                start_date=same_date,
                end_date=same_date,
            )

            assert result["success"] is True


class TestDataConversion:
    """正常系: データ変換のテスト"""

    @pytest.mark.asyncio
    async def test_data_converter_called_correctly(
        self,
        orchestration_service: MarketDataOrchestrationService,
        sample_ohlcv_data: List[Dict[str, Any]],
    ):
        """
        正常系: データコンバーターが正しく呼び出される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_ohlcv_data: サンプルデータ
        """
        orchestration_service.repository.get_latest_ohlcv_data = MagicMock(
            return_value=sample_ohlcv_data
        )

        with patch(
            "app.services.data_collection.orchestration.market_data_orchestration_service.OHLCVDataConverter"
        ) as mock_converter:
            mock_converter.db_to_api_format.return_value = sample_ohlcv_data

            await orchestration_service.get_ohlcv_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                limit=100,
            )

            mock_converter.db_to_api_format.assert_called_once_with(sample_ohlcv_data)


class TestMultipleSymbols:
    """正常系: 複数シンボルのテスト"""

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_different_symbols(
        self,
        orchestration_service: MarketDataOrchestrationService,
        sample_ohlcv_data: List[Dict[str, Any]],
    ):
        """
        正常系: 異なるシンボルでデータが取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            sample_ohlcv_data: サンプルデータ
        """
        symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

        for symbol in symbols:
            orchestration_service.repository.get_latest_ohlcv_data = MagicMock(
                return_value=sample_ohlcv_data
            )

            with patch(
                "app.services.data_collection.orchestration.market_data_orchestration_service.OHLCVDataConverter"
            ) as mock_converter:
                mock_converter.db_to_api_format.return_value = sample_ohlcv_data

                result = await orchestration_service.get_ohlcv_data(
                    symbol=symbol,
                    timeframe="1h",
                    limit=100,
                )

                assert result["success"] is True
                assert result["data"]["symbol"] == symbol




