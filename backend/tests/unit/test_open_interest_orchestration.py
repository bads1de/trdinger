"""
OpenInterestOrchestrationService のテスト

TDDアプローチでOrchestrationServiceの動作を検証します。
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from app.services.data_collection.open_interest_orchestration_service import (
    OpenInterestOrchestrationService,
)


class TestOpenInterestOrchestrationService:
    """OpenInterestOrchestrationService のテストクラス"""

    @pytest.fixture
    def mock_db_session(self):
        """モックデータベースセッション"""
        mock_session = Mock(spec=Session)
        # データベースクエリのモック設定
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        mock_session.query.return_value.distinct.return_value.all.return_value = []
        mock_session.query.return_value.count.return_value = 0
        return mock_session

    @pytest.fixture
    def orchestration_service(self):
        """OrchestrationService インスタンス"""
        return OpenInterestOrchestrationService()

    @pytest.fixture
    def mock_service_result(self):
        """モックサービス結果"""
        return {
            "success": True,
            "saved_count": 100,
            "message": "オープンインタレストデータを 100 件保存しました",
            "symbol": "BTC/USDT:USDT",
        }

    @pytest.mark.asyncio
    async def test_collect_open_interest_data_success(
        self, orchestration_service, mock_db_session, mock_service_result
    ):
        """オープンインタレスト収集の成功テスト"""
        symbol = "BTC/USDT:USDT"
        limit = 200
        fetch_all = False

        with patch(
            "app.services.data_collection.open_interest_orchestration_service.BybitOpenInterestService"
        ) as mock_service_class, patch(
            "app.services.data_collection.open_interest_orchestration_service.OpenInterestRepository"
        ) as mock_repo_class:
            # モック設定
            mock_service = AsyncMock()
            mock_service.fetch_and_save_open_interest_data.return_value = mock_service_result
            mock_service_class.return_value = mock_service

            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # テスト実行
            result = await orchestration_service.collect_open_interest_data(
                symbol=symbol,
                limit=limit,
                fetch_all=fetch_all,
                db_session=mock_db_session,
            )

            # 検証
            assert result["success"] is True
            assert result["data"]["saved_count"] == 100
            assert result["data"]["symbol"] == symbol

            # サービスが正しく呼ばれたことを確認
            mock_service.fetch_and_save_open_interest_data.assert_called_once_with(
                symbol=symbol,
                limit=limit,
                repository=mock_repo,
                fetch_all=fetch_all,
            )

    @pytest.mark.asyncio
    async def test_collect_open_interest_data_failure(
        self, orchestration_service, mock_db_session
    ):
        """オープンインタレスト収集の失敗テスト"""
        symbol = "BTC/USDT:USDT"
        error_result = {
            "success": False,
            "error": "API接続エラー",
            "saved_count": 0,
        }

        with patch(
            "app.services.data_collection.open_interest_orchestration_service.BybitOpenInterestService"
        ) as mock_service_class, patch(
            "app.services.data_collection.open_interest_orchestration_service.OpenInterestRepository"
        ) as mock_repo_class:
            mock_service = AsyncMock()
            mock_service.fetch_and_save_open_interest_data.return_value = error_result
            mock_service_class.return_value = mock_service

            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # テスト実行
            result = await orchestration_service.collect_open_interest_data(
                symbol=symbol, db_session=mock_db_session
            )

            # 検証
            assert result["success"] is False
            assert "API接続エラー" in result["message"]
            assert result["data"]["saved_count"] == 0

    @pytest.mark.asyncio
    async def test_collect_bulk_open_interest_data_success(
        self, orchestration_service, mock_db_session
    ):
        """一括オープンインタレスト収集の成功テスト"""
        symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

        with patch(
            "app.services.data_collection.open_interest_orchestration_service.BybitOpenInterestService"
        ) as mock_service_class, patch(
            "app.services.data_collection.open_interest_orchestration_service.OpenInterestRepository"
        ) as mock_repo_class:
            mock_service = AsyncMock()
            mock_service.fetch_and_save_open_interest_data.return_value = {
                "success": True,
                "saved_count": 50,
            }
            mock_service_class.return_value = mock_service

            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # テスト実行
            result = await orchestration_service.collect_bulk_open_interest_data(
                symbols=symbols, db_session=mock_db_session
            )

            # 検証
            assert result["success"] is True
            assert result["data"]["total_saved"] == 100  # 50 * 2
            assert result["data"]["successful_symbols"] == 2
            assert len(result["data"]["failed_symbols"]) == 0

    @pytest.mark.asyncio
    async def test_get_open_interest_data_success(
        self, orchestration_service, mock_db_session
    ):
        """オープンインタレストデータ取得の成功テスト"""
        symbol = "BTC/USDT:USDT"
        limit = 100

        mock_data = [
            {
                "symbol": "BTC/USDT:USDT",
                "open_interest": 1000.0,
                "timestamp": "2024-01-15T08:00:00Z",
            }
        ]

        with patch(
            "app.services.data_collection.open_interest_orchestration_service.BybitOpenInterestService"
        ) as mock_service_class, patch(
            "app.services.data_collection.open_interest_orchestration_service.OpenInterestRepository"
        ) as mock_repo_class:
            mock_service = Mock()
            mock_service.normalize_symbol.return_value = symbol
            mock_service_class.return_value = mock_service

            mock_repo = Mock()
            mock_repo.get_open_interest_data.return_value = mock_data
            mock_repo_class.return_value = mock_repo

            # テスト実行
            result = await orchestration_service.get_open_interest_data(
                symbol=symbol, limit=limit, db_session=mock_db_session
            )

            # 検証
            assert result["success"] is True
            assert len(result["data"]["open_interest_data"]) == 1
            assert result["data"]["open_interest_data"][0]["symbol"] == symbol

    @pytest.mark.asyncio
    async def test_exception_handling(self, orchestration_service, mock_db_session):
        """例外処理のテスト"""
        symbol = "BTC/USDT:USDT"

        with patch(
            "app.services.data_collection.open_interest_orchestration_service.BybitOpenInterestService"
        ) as mock_service_class:
            # 例外を発生させる
            mock_service_class.side_effect = Exception("接続タイムアウト")

            # テスト実行
            result = await orchestration_service.collect_open_interest_data(
                symbol=symbol, db_session=mock_db_session
            )

            # 検証
            assert result["success"] is False
            assert "接続タイムアウト" in result["message"]

    def test_initialization(self, orchestration_service):
        """初期化テスト"""
        assert orchestration_service is not None
        assert hasattr(orchestration_service, "collect_open_interest_data")
        assert hasattr(orchestration_service, "get_open_interest_data")
        assert hasattr(orchestration_service, "collect_bulk_open_interest_data")


if __name__ == "__main__":
    # 単体でテスト実行
    pytest.main([__file__, "-v"])
