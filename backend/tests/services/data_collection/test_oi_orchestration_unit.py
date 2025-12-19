import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.data_collection.orchestration.open_interest_orchestration_service import OpenInterestOrchestrationService

class TestOpenInterestOrchestrationUnit:
    @pytest.fixture
    def mock_bybit_service(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_bybit_service):
        return OpenInterestOrchestrationService(bybit_service=mock_bybit_service)

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_collect_open_interest_data_success(self, service, mock_bybit_service, mock_db):
        # 正常系: データ収集成功
        mock_bybit_service.fetch_and_save_open_interest_data = AsyncMock(return_value={
            "success": True, "saved_count": 50
        })
        
        with patch("app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository") as mock_repo_cls:
            resp = await service.collect_open_interest_data("BTC/USDT:USDT", fetch_all=True, db_session=mock_db)
            
            assert resp["success"] is True
            assert "50件" in resp["message"]
            assert resp["data"]["saved_count"] == 50
            mock_bybit_service.fetch_and_save_open_interest_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_open_interest_data_failure(self, service, mock_bybit_service, mock_db):
        # 異常系: サービスレベルの失敗
        mock_bybit_service.fetch_and_save_open_interest_data = AsyncMock(return_value={
            "success": False, "error": "API Error", "saved_count": 0
        })
        
        with patch("app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"):
            resp = await service.collect_open_interest_data("BTC/USDT:USDT", db_session=mock_db)
            assert resp["success"] is False
            assert "API Error" in resp["message"]

    @pytest.mark.asyncio
    async def test_get_open_interest_data_normalization(self, service, mock_db):
        # シンボル正規化のテスト
        from datetime import datetime
        mock_record = MagicMock()
        mock_record.symbol = "BTC/USDT:USDT"
        mock_record.open_interest_value = 1000.0
        mock_record.data_timestamp = datetime(2023, 1, 1)
        mock_record.timestamp = datetime(2023, 1, 1)

        with patch("app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository") as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_open_interest_data.return_value = [mock_record]
            
            # Pattern 1: BTC/USDT -> BTC/USDT:USDT
            resp1 = await service.get_open_interest_data("BTC/USDT", db_session=mock_db)
            assert resp1["data"]["symbol"] == "BTC/USDT:USDT"
            
            # Pattern 2: BTC/USD -> BTC/USD:USD (コードロジック確認用)
            # 現在のロジックでは BTC/USD -> BTC/USD:USD ではなく BTC/USD:USDT になる可能性があるので確認
            resp2 = await service.get_open_interest_data("BTC/USD", db_session=mock_db)
            # コード上の elif symbol.endswith("/USD"): f"{symbol}:USD" を確認
            assert resp2["data"]["symbol"] == "BTC/USD:USD"

    @pytest.mark.asyncio
    async def test_collect_bulk_open_interest_data(self, service, mock_bybit_service, mock_db):
        # 一括収集のテスト (一部成功、一部失敗)
        symbols = ["BTC/USDT", "ETH/USDT"]
        
        def side_effect(symbol, **kwargs):
            if "BTC" in symbol:
                return {"success": True, "saved_count": 10}
            else:
                raise Exception("Network Error")
        
        mock_bybit_service.fetch_and_save_open_interest_data = AsyncMock(side_effect=side_effect)
        
        with patch("app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"):
            resp = await service.collect_bulk_open_interest_data(symbols, db_session=mock_db)
            
            assert resp["success"] is True # 全体としては成功（一部失敗含む）
            assert resp["data"]["successful_symbols"] == 1
            assert len(resp["data"]["failed_symbols"]) == 1
            assert resp["data"]["failed_symbols"][0]["symbol"] == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_orchestration_exception_handling(self, service, mock_db):
        # 予期せぬ例外のハンドリング
        with patch.object(service, "_get_db_session", side_effect=Exception("Critical DB Fail")):
            resp = await service.collect_open_interest_data("BTC", db_session=None)
            assert resp["success"] is False
            assert "Critical DB Fail" in resp["details"]["error"]
