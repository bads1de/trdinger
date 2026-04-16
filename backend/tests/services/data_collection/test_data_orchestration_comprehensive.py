from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from app.services.data_collection.orchestration.funding_rate_orchestration_service import (
    FundingRateOrchestrationService,
)
from app.services.data_collection.orchestration.long_short_ratio_orchestration_service import (
    LongShortRatioOrchestrationService,
)
from app.services.data_collection.orchestration.open_interest_orchestration_service import (
    OpenInterestOrchestrationService,
)


class TestDataOrchestrationComprehensive:
    """データ収集オーケストレーションの包括的なテスト (FR, OI, LSR, OHLCV)"""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.fixture
    def mock_bg_tasks(self):
        return MagicMock(spec=BackgroundTasks)

    # ---------------------------------------------------------------------------
    # Funding Rate Orchestration
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_funding_rate_orchestration(self, mock_db):
        mock_service = MagicMock()
        mock_service.fetch_funding_rate_history = AsyncMock(
            return_value=[{"timestamp": 1000, "rate": 0.01}]
        )
        mock_service.fetch_all_funding_rate_history = AsyncMock(
            return_value=[{"timestamp": 1000, "rate": 0.01}]
        )

        with patch(
            "app.services.data_collection.orchestration.funding_rate_orchestration_service.FundingRateRepository"
        ) as MockRepo:
            MockRepo.return_value.insert_funding_rate_data.return_value = 10
            service = FundingRateOrchestrationService(bybit_service=mock_service)
            result = await service.collect_funding_rate_data(
                "BTCUSDT", limit=100, fetch_all=True, db_session=mock_db
            )

            assert result["success"] is True
            assert result["data"]["count"] == 10

    # ---------------------------------------------------------------------------
    # Open Interest Orchestration
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_open_interest_orchestration(self, mock_db):
        mock_service = MagicMock()
        mock_service.fetch_and_save_open_interest_data = AsyncMock(
            return_value={"success": True, "saved_count": 10}
        )

        service = OpenInterestOrchestrationService(bybit_service=mock_service)
        result = await service.collect_open_interest_data("BTCUSDT", db_session=mock_db)

        assert result["success"] is True
        assert result["data"]["saved_count"] == 10

    # ---------------------------------------------------------------------------
    # Long/Short Ratio Orchestration
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_long_short_ratio_orchestration(self, mock_db):
        mock_service = MagicMock()
        mock_service.collect_historical_long_short_ratio_data = AsyncMock(
            return_value=50
        )

        service = LongShortRatioOrchestrationService(bybit_service=mock_service)
        result = await service.collect_long_short_ratio_data(
            "BTCUSDT", period="1h", fetch_all=True, db_session=mock_db
        )

        assert result["success"] is True
        assert result["data"]["count"] == 50

    # ---------------------------------------------------------------------------
    # Data Collection (OHLCV) Orchestration Unit
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_start_historical_collection(self, mock_db, mock_bg_tasks):
        service = DataCollectionOrchestrationService()

        with patch.object(
            service, "validate_symbol_and_timeframe", return_value="BTC/USDT:USDT"
        ):
            with patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ) as MockRepo:
                MockRepo.return_value.get_data_count.return_value = 0

                resp = await service.start_historical_data_collection(
                    "BTC/USDT:USDT", "1h", mock_bg_tasks, mock_db
                )
                assert resp["status"] == "started"
                mock_bg_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_incremental_update(self, mock_db):
        service = DataCollectionOrchestrationService()
        with patch.object(
            service.historical_service,
            "collect_bulk_incremental_data",
            AsyncMock(return_value={"ohlcv": 10}),
        ):
            resp = await service.execute_bulk_incremental_update(
                "BTC/USDT:USDT", mock_db
            )
            assert resp["success"] is True
            assert resp["data"]["ohlcv"] == 10

    def test_parse_datetime_utility(self):
        service = FundingRateOrchestrationService(bybit_service=MagicMock())
        dt = service._parse_datetime("2023-01-01T00:00:00")
        assert isinstance(dt, datetime)
        assert service._parse_datetime(None) is None
