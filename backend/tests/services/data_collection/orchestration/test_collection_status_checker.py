"""
data_collection/orchestration/collection_status_checker モジュールのユニットテスト
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.data_collection.orchestration.collection_status_checker import (
    CollectionStatusChecker,
)


@pytest.fixture
def checker():
    return CollectionStatusChecker()


class TestCollectionStatusChecker:
    @pytest.mark.asyncio
    @patch(
        "app.services.data_collection.orchestration.collection_status_checker.OHLCVRepository"
    )
    @patch("app.config.unified_config.unified_config")
    async def test_get_collection_status_no_data(
        self, mock_config, mock_repo_class, checker
    ):
        mock_config.market.symbol_mapping = {}
        mock_config.market.supported_symbols = ["BTCUSDT"]
        mock_config.market.supported_timeframes = ["1h"]

        mock_repo = MagicMock()
        mock_repo.get_data_count.return_value = 0
        mock_repo_class.return_value = mock_repo

        mock_db = MagicMock()
        mock_bg = MagicMock()

        result = await checker.get_collection_status(
            symbol="BTCUSDT",
            timeframe="1h",
            background_tasks=mock_bg,
            auto_fetch=False,
            db=mock_db,
        )

        assert result["success"] is True
        assert result["status"] == "no_data"

    @pytest.mark.asyncio
    @patch(
        "app.services.data_collection.orchestration.collection_status_checker.OHLCVRepository"
    )
    @patch("app.config.unified_config.unified_config")
    async def test_get_collection_status_data_exists(
        self, mock_config, mock_repo_class, checker
    ):
        mock_config.market.symbol_mapping = {}
        mock_config.market.supported_symbols = ["BTCUSDT"]
        mock_config.market.supported_timeframes = ["1h"]

        from datetime import datetime, timezone

        mock_repo = MagicMock()
        mock_repo.get_data_count.return_value = 1000
        mock_repo.get_latest_timestamp.return_value = datetime(
            2024, 1, 10, tzinfo=timezone.utc
        )
        mock_repo.get_oldest_timestamp.return_value = datetime(
            2024, 1, 1, tzinfo=timezone.utc
        )
        mock_repo_class.return_value = mock_repo

        mock_db = MagicMock()
        mock_bg = MagicMock()

        result = await checker.get_collection_status(
            symbol="BTCUSDT",
            timeframe="1h",
            background_tasks=mock_bg,
            auto_fetch=False,
            db=mock_db,
        )

        assert result["success"] is True
        assert result["data"]["data_count"] == 1000
        assert result["data"]["status"] == "data_exists"

    @pytest.mark.asyncio
    @patch(
        "app.services.data_collection.orchestration.collection_status_checker.OHLCVRepository"
    )
    async def test_get_collection_status_with_data_validator(
        self, mock_repo_class, checker
    ):
        mock_data_validator = MagicMock()
        mock_data_validator.validate_symbol_and_timeframe.return_value = "BTCUSDT"

        mock_repo = MagicMock()
        mock_repo.get_data_count.return_value = 500
        mock_repo.get_latest_timestamp.return_value = None
        mock_repo.get_oldest_timestamp.return_value = None
        mock_repo_class.return_value = mock_repo

        result = await checker.get_collection_status(
            symbol="BTCUSDT",
            timeframe="1h",
            background_tasks=MagicMock(),
            auto_fetch=False,
            db=MagicMock(),
            data_validator=mock_data_validator,
        )

        assert result["success"] is True
        mock_data_validator.validate_symbol_and_timeframe.assert_called_once_with(
            "BTCUSDT", "1h"
        )
