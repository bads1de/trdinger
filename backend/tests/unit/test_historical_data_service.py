"""
履歴データ収集サービスの簡素なテスト
"""
import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone

from app.core.services.historical_data_service import HistoricalDataService


class TestHistoricalDataService:
    """履歴データ収集サービスのテスト"""

    @pytest.fixture
    def mock_market_service(self):
        """モック市場データサービス"""
        service = Mock()
        service.fetch_ohlcv_data = AsyncMock()
        return service

    @pytest.fixture
    def mock_repository(self):
        """モックリポジトリ"""
        repo = Mock()
        repo.get_latest_timestamp = Mock()
        repo.insert_ohlcv_data = Mock()
        return repo

    @pytest.fixture
    def service(self, mock_market_service):
        """テスト用サービス"""
        return HistoricalDataService(mock_market_service)

    @pytest.mark.asyncio
    async def test_collect_historical_data_success(self, service, mock_market_service, mock_repository):
        """履歴データ収集の成功テスト"""
        # Given
        mock_market_service.fetch_ohlcv_data.return_value = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0]
        ]
        mock_repository.insert_ohlcv_data.return_value = 1

        # When
        result = await service.collect_historical_data("BTC/USDT", "1h", mock_repository)

        # Then
        assert result["success"] is True
        assert result["saved_count"] > 0
        mock_market_service.fetch_ohlcv_data.assert_called()

    @pytest.mark.asyncio
    async def test_collect_incremental_data_success(self, service, mock_market_service, mock_repository):
        """差分データ収集の成功テスト"""
        # Given
        mock_repository.get_latest_timestamp.return_value = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_market_service.fetch_ohlcv_data.return_value = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0]
        ]
        mock_repository.insert_ohlcv_data.return_value = 1

        # When
        result = await service.collect_incremental_data("BTC/USDT", "1h", mock_repository)

        # Then
        assert result["success"] is True
        mock_repository.get_latest_timestamp.assert_called_with("BTC/USDT", "1h")
