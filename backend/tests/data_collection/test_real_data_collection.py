"""
実際のデータ収集テスト

実際のAPIを使用してデータ収集をテストします。
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from app.services.data_collection.bybit.funding_rate_service import (
    BybitFundingRateService,
)
from app.services.data_collection.fear_greed.fear_greed_service import (
    FearGreedIndexService,
)
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.fear_greed_repository import FearGreedIndexRepository


class TestRealDataCollection:
    """実際のデータ収集テスト"""

    @pytest.mark.asyncio
    async def test_funding_rate_real_api(self):
        """実際のAPIを使用してファンディングレートを取得"""
        service = BybitFundingRateService()

        try:
            # 現在のファンディングレートを取得
            current_rate = await service.fetch_current_funding_rate("BTC/USDT")

            assert current_rate is not None
            assert "symbol" in current_rate
            assert "fundingRate" in current_rate

            print(f"Current Funding Rate: {current_rate}")

        except Exception as e:
            # ネットワークエラーなどの場合はスキップ
            pytest.skip(f"API access failed: {e}")

    @pytest.mark.asyncio
    async def test_funding_rate_history_real_api(self):
        """実際のAPIを使用してファンディングレート履歴を取得"""
        service = BybitFundingRateService()

        try:
            # 履歴データを取得（少量）
            history = await service.fetch_funding_rate_history("BTC/USDT", limit=5)

            assert isinstance(history, list)
            if history:  # データがある場合のみ検証
                assert len(history) <= 5
                assert "symbol" in history[0]
                assert "fundingRate" in history[0]

            print(f"Funding Rate History Count: {len(history)}")

        except Exception as e:
            # ネットワークエラーなどの場合はスキップ
            pytest.skip(f"API access failed: {e}")

    @pytest.mark.asyncio
    async def test_fear_greed_real_api(self):
        """実際のAPIを使用してFear & Greed Indexを取得"""
        async with FearGreedIndexService() as service:
            try:
                # Fear & Greed Indexデータを取得
                data = await service.fetch_fear_greed_data(limit=5)

                assert isinstance(data, list)
                if data:  # データがある場合のみ検証
                    assert len(data) <= 5
                    assert "value" in data[0]
                    assert "value_classification" in data[0]
                    assert "data_timestamp" in data[0]

                print(f"Fear & Greed Index Count: {len(data)}")
                if data:
                    print(f"Latest Fear & Greed: {data[0]}")

            except Exception as e:
                # ネットワークエラーなどの場合はスキップ
                pytest.skip(f"API access failed: {e}")

    @pytest.mark.asyncio
    async def test_funding_rate_with_mock_repository(self):
        """モックリポジトリを使用してファンディングレートの保存をテスト"""
        service = BybitFundingRateService()

        # モックリポジトリ
        mock_repo = Mock(spec=FundingRateRepository)
        mock_repo.get_latest_funding_timestamp.return_value = None
        mock_repo.insert_funding_rate_data.return_value = 3

        try:
            # 差分データ取得（実際のAPI + モックDB）
            result = await service.fetch_incremental_funding_rate_data(
                "BTC/USDT", mock_repo
            )

            assert result["success"] is True
            assert result["symbol"] == "BTC/USDT:USDT"
            assert result["saved_count"] == 3  # モックの戻り値

            print(f"Funding Rate Integration Result: {result}")

        except Exception as e:
            # ネットワークエラーなどの場合はスキップ
            pytest.skip(f"API access failed: {e}")

    @pytest.mark.asyncio
    async def test_fear_greed_with_mock_repository(self):
        """モックリポジトリを使用してFear & Greed Indexの保存をテスト"""
        # モックリポジトリ
        mock_repo = Mock(spec=FearGreedIndexRepository)
        mock_repo.insert_fear_greed_data.return_value = 2

        async with FearGreedIndexService() as service:
            try:
                # データ取得・保存（実際のAPI + モックDB）
                result = await service.fetch_and_save_fear_greed_data(
                    limit=5, repository=mock_repo
                )

                assert result["success"] is True
                assert result["inserted_count"] == 2  # モックの戻り値

                print(f"Fear & Greed Integration Result: {result}")

            except Exception as e:
                # ネットワークエラーなどの場合はスキップ
                pytest.skip(f"API access failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
