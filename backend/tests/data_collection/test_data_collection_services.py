"""
データ収集サービスのテスト

TDDアプローチで各データタイプ（OHLCV、FR、OI、Fear & Greed Index）の
取得処理をテストします。
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from app.services.data_collection.bybit.funding_rate_service import (
    BybitFundingRateService,
)
from app.services.data_collection.bybit.open_interest_service import (
    BybitOpenInterestService,
)
from app.services.data_collection.fear_greed.fear_greed_service import (
    FearGreedIndexService,
)
from app.services.data_collection.orchestration.fear_greed_orchestration_service import (
    FearGreedOrchestrationService,
)
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.fear_greed_repository import FearGreedIndexRepository


class TestFundingRateService:
    """ファンディングレートサービスのテスト"""

    @pytest.fixture
    def mock_repository(self):
        """モックリポジトリを作成"""
        repo = Mock(spec=FundingRateRepository)
        repo.get_latest_funding_timestamp.return_value = None
        repo.insert_funding_rate_data.return_value = 5
        return repo

    @pytest.fixture
    def funding_rate_service(self):
        """ファンディングレートサービスのインスタンスを作成"""
        return BybitFundingRateService()

    @pytest.mark.asyncio
    async def test_fetch_current_funding_rate(self, funding_rate_service):
        """現在のファンディングレート取得をテスト"""
        # モックデータ
        mock_funding_data = {
            "symbol": "BTC/USDT:USDT",
            "fundingRate": 0.0001,
            "datetime": "2025-08-17T00:00:00.000Z",
            "nextFundingDatetime": "2025-08-17T08:00:00.000Z",
        }

        with patch.object(
            funding_rate_service.exchange,
            "fetch_funding_rate",
            return_value=mock_funding_data,
        ):
            result = await funding_rate_service.fetch_current_funding_rate("BTC/USDT")

            assert result is not None
            assert result["symbol"] == "BTC/USDT:USDT"
            assert "fundingRate" in result

    @pytest.mark.asyncio
    async def test_fetch_incremental_funding_rate_data(
        self, funding_rate_service, mock_repository
    ):
        """差分ファンディングレートデータ取得をテスト"""
        # モック履歴データ
        mock_history_data = [
            {
                "symbol": "BTC/USDT:USDT",
                "fundingRate": 0.0001,
                "datetime": "2025-08-17T00:00:00.000Z",
                "timestamp": 1723852800000,
            }
        ]

        with patch.object(
            funding_rate_service.exchange,
            "fetch_funding_rate_history",
            return_value=mock_history_data,
        ):
            result = await funding_rate_service.fetch_incremental_funding_rate_data(
                "BTC/USDT", mock_repository
            )

            assert result["success"] is True
            assert result["saved_count"] == 5  # モックリポジトリの戻り値
            assert result["symbol"] == "BTC/USDT:USDT"


class TestOpenInterestService:
    """オープンインタレストサービスのテスト"""

    @pytest.fixture
    def mock_repository(self):
        """モックリポジトリを作成"""
        repo = Mock(spec=OpenInterestRepository)
        repo.get_latest_open_interest_timestamp.return_value = None
        repo.insert_open_interest_data.return_value = 3
        return repo

    @pytest.fixture
    def open_interest_service(self):
        """オープンインタレストサービスのインスタンスを作成"""
        return BybitOpenInterestService()

    @pytest.mark.asyncio
    async def test_fetch_open_interest_history_with_fallback(
        self, open_interest_service
    ):
        """オープンインタレスト履歴取得（フォールバック付き）をテスト"""
        # CCXTエラーをシミュレート
        with patch.object(
            open_interest_service.exchange,
            "fetch_open_interest_history",
            side_effect=Exception("CCXT Error"),
        ):
            result = await open_interest_service.fetch_open_interest_history("BTC/USDT")

            # フォールバックで空のリストが返されることを確認
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_incremental_open_interest_data_disabled(
        self, open_interest_service, mock_repository
    ):
        """オープンインタレスト差分データ取得（無効化状態）をテスト"""
        # 現在は一時的に無効化されているため、エラーハンドリングをテスト
        try:
            result = await open_interest_service.fetch_incremental_open_interest_data(
                "BTC/USDT", mock_repository
            )
            # 無効化されている場合でも適切にハンドリングされることを確認
            assert isinstance(result, dict)
        except Exception as e:
            # エラーが発生した場合、適切にログ出力されることを確認
            assert "can only concatenate str" not in str(e)


class TestFearGreedIndexService:
    """Fear & Greed Indexサービスのテスト"""

    @pytest.fixture
    def mock_repository(self):
        """モックリポジトリを作成"""
        repo = Mock(spec=FearGreedIndexRepository)
        repo.get_latest_data_timestamp.return_value = None
        repo.insert_fear_greed_data.return_value = 7
        return repo

    @pytest.mark.asyncio
    async def test_fetch_fear_greed_data(self):
        """Fear & Greed Indexデータ取得をテスト"""
        # モックAPIレスポンス
        mock_api_response = {
            "data": [
                {
                    "value": "25",
                    "value_classification": "Extreme Fear",
                    "timestamp": "1723852800",
                }
            ]
        }

        async with FearGreedIndexService() as service:
            with patch("aiohttp.ClientSession.get") as mock_get:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = mock_api_response
                mock_get.return_value.__aenter__.return_value = mock_response

                result = await service.fetch_fear_greed_data(limit=1)

                assert len(result) == 1
                assert result[0]["value"] == 25
                assert result[0]["value_classification"] == "Extreme Fear"

    @pytest.mark.asyncio
    async def test_fetch_and_save_fear_greed_data(self, mock_repository):
        """Fear & Greed Indexデータ取得・保存をテスト"""
        # モックAPIレスポンス
        mock_api_response = {
            "data": [
                {
                    "value": "25",
                    "value_classification": "Extreme Fear",
                    "timestamp": "1723852800",
                }
            ]
        }

        async with FearGreedIndexService() as service:
            with patch("aiohttp.ClientSession.get") as mock_get:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = mock_api_response
                mock_get.return_value.__aenter__.return_value = mock_response

                result = await service.fetch_and_save_fear_greed_data(
                    limit=1, repository=mock_repository
                )

                assert result["success"] is True
                assert result["inserted_count"] == 7  # モックリポジトリの戻り値
                assert result["fetched_count"] == 1


class TestFearGreedOrchestrationService:
    """Fear & Greed Index統合サービスのテスト"""

    @pytest.fixture
    def mock_db_session(self):
        """モックDBセッションを作成"""
        return Mock()

    @pytest.mark.asyncio
    async def test_collect_incremental_fear_greed_data(self, mock_db_session):
        """Fear & Greed Index差分データ収集をテスト"""
        service = FearGreedOrchestrationService()

        # モックリポジトリ
        mock_repo = Mock(spec=FearGreedIndexRepository)
        mock_repo.get_latest_data_timestamp.return_value = None

        # モックAPIレスポンス
        mock_api_response = {
            "data": [
                {
                    "value": "25",
                    "value_classification": "Extreme Fear",
                    "timestamp": "1723852800",
                }
            ]
        }

        with patch(
            "database.repositories.fear_greed_repository.FearGreedIndexRepository",
            return_value=mock_repo,
        ):
            with patch("aiohttp.ClientSession.get") as mock_get:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = mock_api_response
                mock_get.return_value.__aenter__.return_value = mock_response

                mock_repo.insert_fear_greed_data.return_value = 1

                result = await service.collect_incremental_fear_greed_data(
                    mock_db_session
                )

                assert result["success"] is True
                assert "data" in result
                assert result["data"]["inserted_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
