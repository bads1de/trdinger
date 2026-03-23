from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.data_collection.orchestration.long_short_ratio_orchestration_service import (
    LongShortRatioOrchestrationService,
)


@pytest.fixture
def mock_db_session() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_bybit_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def orchestration_service(
    mock_bybit_service: AsyncMock,
) -> LongShortRatioOrchestrationService:
    return LongShortRatioOrchestrationService(bybit_service=mock_bybit_service)


class TestCollectLongShortRatioData:
    @pytest.mark.asyncio
    async def test_collect_long_short_ratio_data_success(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ) -> None:
        orchestration_service.bybit_service.fetch_incremental_long_short_ratio_data = (
            AsyncMock(return_value={"saved_count": 12})
        )

        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ):
            result = await orchestration_service.collect_long_short_ratio_data(
                symbol="BTC/USDT:USDT",
                period="1h",
                fetch_all=False,
                db_session=mock_db_session,
            )

        assert result["success"] is True
        assert result["data"]["count"] == 12

    @pytest.mark.asyncio
    async def test_collect_long_short_ratio_data_error_returns_error_response(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ) -> None:
        orchestration_service.bybit_service.fetch_incremental_long_short_ratio_data = (
            AsyncMock(side_effect=Exception("API error"))
        )

        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ):
            result = await orchestration_service.collect_long_short_ratio_data(
                symbol="BTC/USDT:USDT",
                period="1h",
                fetch_all=False,
                db_session=mock_db_session,
            )

        assert result["success"] is False
        assert "API error" in result["message"]
