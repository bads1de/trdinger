"""
データ収集統合テスト

実際のデータ収集処理をテストして問題を特定します。
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository


class TestDataCollectionIntegration:
    """データ収集統合テスト"""

    @pytest.fixture
    def mock_db_session(self):
        """モックDBセッションを作成"""
        return Mock(spec=Session)

    @pytest.fixture
    def orchestration_service(self):
        """データ収集統合サービスのインスタンスを作成"""
        return DataCollectionOrchestrationService()

    @pytest.fixture
    def mock_repositories(self, mock_db_session):
        """モックリポジトリを作成"""
        ohlcv_repo = Mock(spec=OHLCVRepository)
        funding_rate_repo = Mock(spec=FundingRateRepository)
        open_interest_repo = Mock(spec=OpenInterestRepository)

        # OHLCVリポジトリの設定
        ohlcv_repo.get_data_count.return_value = 100  # 既存データがある

        # ファンディングレートリポジトリの設定
        funding_rate_repo.get_latest_funding_timestamp.return_value = None
        funding_rate_repo.insert_funding_rate_data.return_value = 5

        # オープンインタレストリポジトリの設定
        open_interest_repo.get_latest_open_interest_timestamp.return_value = None
        open_interest_repo.insert_open_interest_data.return_value = 3

        return {
            "ohlcv": ohlcv_repo,
            "funding_rate": funding_rate_repo,
            "open_interest": open_interest_repo,
        }

    @pytest.mark.asyncio
    async def test_bulk_incremental_update_funding_rate_only(
        self, orchestration_service, mock_db_session, mock_repositories
    ):
        """ファンディングレートのみの一括差分更新をテスト"""

        # モックファンディングレート履歴データ
        mock_funding_history = [
            {
                "symbol": "BTC/USDT:USDT",
                "fundingRate": 0.0001,
                "datetime": "2025-08-17T00:00:00.000Z",
                "timestamp": 1723852800000,
            }
        ]

        with patch(
            "database.repositories.ohlcv_repository.OHLCVRepository",
            return_value=mock_repositories["ohlcv"],
        ):
            with patch(
                "database.repositories.funding_rate_repository.FundingRateRepository",
                return_value=mock_repositories["funding_rate"],
            ):
                with patch(
                    "database.repositories.open_interest_repository.OpenInterestRepository",
                    return_value=mock_repositories["open_interest"],
                ):
                    with patch(
                        "app.services.data_collection.bybit.funding_rate_service.BybitFundingRateService"
                    ) as mock_fr_service_class:

                        # ファンディングレートサービスのモック設定
                        mock_fr_service = Mock()
                        mock_fr_service.fetch_incremental_funding_rate_data.return_value = {
                            "symbol": "BTC/USDT:USDT",
                            "saved_count": 5,
                            "success": True,
                            "latest_timestamp": None,
                        }
                        mock_fr_service_class.return_value = mock_fr_service

                        # 一括差分更新を実行
                        result = (
                            await orchestration_service.execute_bulk_incremental_update(
                                symbol="BTC/USDT:USDT", db=mock_db_session
                            )
                        )

                        # 結果を検証
                        assert result["success"] is True
                        assert "data" in result

                        # ファンディングレートデータが処理されたことを確認
                        funding_rate_data = result["data"].get("funding_rate")
                        if funding_rate_data:
                            print(f"Funding Rate Result: {funding_rate_data}")
                            assert funding_rate_data["success"] is True
                            assert funding_rate_data["saved_count"] == 5
                        else:
                            print("Funding Rate data not found in result")
                            print(f"Available data keys: {list(result['data'].keys())}")

    @pytest.mark.asyncio
    async def test_historical_service_funding_rate_collection(
        self, mock_db_session, mock_repositories
    ):
        """履歴サービスのファンディングレート収集を直接テスト"""
        from app.services.data_collection.historical.historical_data_service import (
            HistoricalDataService,
        )

        historical_service = HistoricalDataService()

        with patch(
            "app.services.data_collection.bybit.funding_rate_service.BybitFundingRateService"
        ) as mock_fr_service_class:
            # ファンディングレートサービスのモック設定
            mock_fr_service = Mock()
            # 非同期メソッドなのでAsyncMockを使用
            from unittest.mock import AsyncMock

            mock_fr_service.fetch_incremental_funding_rate_data = AsyncMock(
                return_value={
                    "symbol": "BTC/USDT:USDT",
                    "saved_count": 5,
                    "success": True,
                    "latest_timestamp": None,
                }
            )
            mock_fr_service_class.return_value = mock_fr_service

            # 一括差分データ収集を実行
            result = await historical_service.collect_bulk_incremental_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                ohlcv_repository=mock_repositories["ohlcv"],
                funding_rate_repository=mock_repositories["funding_rate"],
                open_interest_repository=mock_repositories["open_interest"],
            )

            # 結果を検証
            print(f"Historical Service Result: {result}")

            # ファンディングレートが処理されたことを確認
            funding_rate_data = result["data"].get("funding_rate")
            assert (
                funding_rate_data is not None
            ), f"Funding rate data not found. Available keys: {list(result['data'].keys())}"
            assert funding_rate_data["success"] is True
            assert funding_rate_data["saved_count"] == 5

    @pytest.mark.asyncio
    async def test_funding_rate_service_direct(self):
        """ファンディングレートサービスを直接テスト"""
        from app.services.data_collection.bybit.funding_rate_service import (
            BybitFundingRateService,
        )

        service = BybitFundingRateService()

        # モックリポジトリ
        mock_repo = Mock()
        mock_repo.get_latest_funding_timestamp.return_value = None
        mock_repo.insert_funding_rate_data.return_value = 5

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
            service.exchange,
            "fetch_funding_rate_history",
            return_value=mock_history_data,
        ):
            result = await service.fetch_incremental_funding_rate_data(
                "BTC/USDT", mock_repo
            )

            print(f"Direct Funding Rate Service Result: {result}")
            assert result["success"] is True
            assert result["saved_count"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
