"""
ExternalMarketOrchestrationService のテスト

TDDアプローチでOrchestrationServiceの動作を検証します。
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.core.services.data_collection.external_market_orchestration_service import (
    ExternalMarketOrchestrationService,
)


class TestExternalMarketOrchestrationService:
    """ExternalMarketOrchestrationService のテストクラス"""

    @pytest.fixture
    def mock_db_session(self):
        """モックデータベースセッション"""
        mock_session = Mock(spec=Session)
        # データベースクエリのモック設定
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
            None
        )
        mock_session.query.return_value.distinct.return_value.all.return_value = []
        mock_session.query.return_value.count.return_value = 0
        return mock_session

    @pytest.fixture
    def orchestration_service(self):
        """OrchestrationService インスタンス"""
        return ExternalMarketOrchestrationService()

    @pytest.fixture
    def mock_collector_result(self):
        """モックコレクター結果"""
        return {
            "success": True,
            "message": "外部市場データを 10 件保存しました",
            "fetched_count": 10,
            "inserted_count": 10,
            "collection_type": "incremental",
        }

    @pytest.mark.asyncio
    async def test_collect_incremental_data_success(
        self, orchestration_service, mock_db_session, mock_collector_result
    ):
        """差分データ収集の成功テスト"""
        symbols = ["^GSPC", "^IXIC"]

        with patch(
            "app.core.services.data_collection.external_market_orchestration_service.ExternalMarketDataCollector"
        ) as mock_collector_class:
            # AsyncContextManagerのモック設定
            mock_collector = AsyncMock()
            mock_collector.collect_incremental_external_market_data.return_value = (
                mock_collector_result
            )
            mock_collector_class.return_value.__aenter__.return_value = mock_collector

            # テスト実行
            result = await orchestration_service.collect_incremental_data(
                symbols=symbols, db_session=mock_db_session
            )

            # 検証
            assert result["success"] is True
            assert result["message"] == mock_collector_result["message"]
            assert result["data"]["fetched_count"] == 10
            assert result["data"]["inserted_count"] == 10

            # コレクターが正しく呼ばれたことを確認
            mock_collector.collect_incremental_external_market_data.assert_called_once_with(
                symbols=symbols, db_session=mock_db_session
            )

    @pytest.mark.asyncio
    async def test_collect_incremental_data_failure(
        self, orchestration_service, mock_db_session
    ):
        """差分データ収集の失敗テスト"""
        symbols = ["^GSPC"]
        error_result = {
            "success": False,
            "error": "API接続エラー",
            "fetched_count": 0,
            "inserted_count": 0,
        }

        with patch(
            "app.core.services.data_collection.external_market_orchestration_service.ExternalMarketDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.collect_incremental_external_market_data.return_value = (
                error_result
            )
            mock_collector_class.return_value.__aenter__.return_value = mock_collector

            # テスト実行
            result = await orchestration_service.collect_incremental_data(
                symbols=symbols, db_session=mock_db_session
            )

            # 検証
            assert result["success"] is False
            assert "API接続エラー" in result["message"]
            assert result["data"]["fetched_count"] == 0

    @pytest.mark.asyncio
    async def test_get_data_status_success(
        self, orchestration_service, mock_db_session
    ):
        """データ状態取得の成功テスト"""
        status_result = {
            "success": True,
            "total_records": 1000,
            "symbols": ["^GSPC", "^IXIC", "DX-Y.NYB", "^VIX"],
            "latest_timestamp": "2024-01-15T10:00:00Z",
            "oldest_timestamp": "2023-01-01T00:00:00Z",
        }

        with patch(
            "app.core.services.data_collection.external_market_orchestration_service.ExternalMarketDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.get_external_market_data_status.return_value = status_result
            mock_collector_class.return_value.__aenter__.return_value = mock_collector

            # テスト実行
            result = await orchestration_service.get_data_status(
                db_session=mock_db_session
            )

            # 検証
            assert result["success"] is True
            assert result["data"]["total_records"] == 1000
            assert len(result["data"]["symbols"]) == 4

    @pytest.mark.asyncio
    async def test_collect_historical_data_success(
        self, orchestration_service, mock_db_session
    ):
        """履歴データ収集の成功テスト"""
        symbols = ["^GSPC"]
        period = "1y"

        historical_result = {
            "success": True,
            "message": "履歴データを 365 件保存しました",
            "fetched_count": 365,
            "inserted_count": 365,
        }

        with patch(
            "app.core.services.data_collection.external_market_orchestration_service.ExternalMarketDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.collect_external_market_data.return_value = historical_result
            mock_collector_class.return_value.__aenter__.return_value = mock_collector

            # テスト実行
            result = await orchestration_service.collect_historical_data(
                symbols=symbols, period=period, db_session=mock_db_session
            )

            # 検証
            assert result["success"] is True
            assert result["data"]["fetched_count"] == 365
            assert result["data"]["inserted_count"] == 365

            # コレクターが正しく呼ばれたことを確認
            mock_collector.collect_external_market_data.assert_called_once_with(
                symbols=symbols, period=period, db_session=mock_db_session
            )

    @pytest.mark.asyncio
    async def test_exception_handling(self, orchestration_service, mock_db_session):
        """例外処理のテスト"""
        symbols = ["^GSPC"]

        with patch(
            "app.core.services.data_collection.external_market_orchestration_service.ExternalMarketDataCollector"
        ) as mock_collector_class:
            # 例外を発生させる
            mock_collector_class.side_effect = Exception("接続タイムアウト")

            # テスト実行
            result = await orchestration_service.collect_incremental_data(
                symbols=symbols, db_session=mock_db_session
            )

            # 検証
            assert result["success"] is False
            assert "接続タイムアウト" in result["message"]

    def test_initialization(self, orchestration_service):
        """初期化テスト"""
        assert orchestration_service is not None
        assert hasattr(orchestration_service, "collect_incremental_data")
        assert hasattr(orchestration_service, "get_data_status")
        assert hasattr(orchestration_service, "collect_historical_data")


if __name__ == "__main__":
    # 単体でテスト実行
    pytest.main([__file__, "-v"])
