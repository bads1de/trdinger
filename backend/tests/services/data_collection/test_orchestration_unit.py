import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import BackgroundTasks
from app.services.data_collection.orchestration.data_collection_orchestration_service import DataCollectionOrchestrationService

class TestDataCollectionOrchestrationUnit:
    @pytest.fixture
    def service(self):
        return DataCollectionOrchestrationService()

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.fixture
    def mock_bg_tasks(self):
        return MagicMock(spec=BackgroundTasks)

    def test_validate_symbol_and_timeframe(self, service):
        # 1. 正常系
        with patch("app.config.unified_config.unified_config.market.supported_symbols", ["BTC/USDT:USDT"]):
            with patch("app.config.unified_config.unified_config.market.supported_timeframes", ["1h"]):
                res = service.validate_symbol_and_timeframe("BTC/USDT:USDT", "1h")
                assert res == "BTC/USDT:USDT"
                
                # 正規化のテスト (symbol_mapping)
                with patch("app.config.unified_config.unified_config.market.symbol_mapping", {"BTC": "BTC/USDT:USDT"}):
                    assert service.validate_symbol_and_timeframe("BTC", "1h") == "BTC/USDT:USDT"

    @pytest.mark.asyncio
    async def test_start_historical_data_collection_force(self, service, mock_db, mock_bg_tasks):
        # 強制更新モード
        with patch.object(service, "validate_symbol_and_timeframe", return_value="BTC/USDT:USDT"):
            with patch("app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository") as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                mock_repo.get_data_count.return_value = 100
                
                resp = await service.start_historical_data_collection(
                    "BTC/USDT:USDT", "1h", mock_bg_tasks, mock_db, force_update=True
                )
                assert resp["success"] is True
                assert "強制更新" in resp["message"]
                mock_repo.clear_ohlcv_data_by_symbol_and_timeframe.assert_called_once()


    @pytest.mark.asyncio
    async def test_start_historical_data_collection(self, service, mock_db, mock_bg_tasks):
        # 正常系: 新規収集開始
        with patch.object(service, "validate_symbol_and_timeframe", return_value="BTC/USDT:USDT"):
            with patch("app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository") as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                mock_repo.get_data_count.return_value = 0 # データなし
                
                resp = await service.start_historical_data_collection(
                    "BTC/USDT:USDT", "1h", mock_bg_tasks, mock_db
                )
                
                assert resp["success"] is True
                assert resp["status"] == "started"
                # バックグラウンドタスクが追加されたか確認
                mock_bg_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_historical_data_collection_exists(self, service, mock_db, mock_bg_tasks):
        # 正常系: 既にデータが存在する場合 (新規収集しない)
        with patch.object(service, "validate_symbol_and_timeframe", return_value="BTC/USDT:USDT"):
            with patch("app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository") as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                mock_repo.get_data_count.return_value = 100 # データあり
                
                resp = await service.start_historical_data_collection(
                    "BTC/USDT:USDT", "1h", mock_bg_tasks, mock_db, force_update=False
                )
                
                assert resp["success"] is True
                assert resp["status"] == "exists"
                mock_bg_tasks.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_bulk_incremental_update(self, service, mock_db):
        # 一括差分更新
        with patch.object(service.historical_service, "collect_bulk_incremental_data", new_callable=AsyncMock) as mock_collect:
            mock_collect.return_value = {"ohlcv": 10}
            
            resp = await service.execute_bulk_incremental_update("BTC/USDT:USDT", mock_db)
            assert resp["success"] is True
            assert resp["data"] == {"ohlcv": 10}

    @pytest.mark.asyncio
    async def test_get_collection_status_no_data(self, service, mock_db, mock_bg_tasks):
        # データがない場合のステータス確認
        with patch("app.config.unified_config.unified_config.market.supported_symbols", ["BTC/USDT:USDT"]):
            with patch("app.config.unified_config.unified_config.market.supported_timeframes", ["1h"]):
                with patch("app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository") as mock_repo_cls:
                    mock_repo = mock_repo_cls.return_value
                    mock_repo.get_data_count.return_value = 0
                    
                    # auto_fetch=False
                    resp = await service.get_collection_status("BTC/USDT:USDT", "1h", mock_bg_tasks, False, mock_db)
                    assert resp["status"] == "no_data"
                    
                    # auto_fetch=True
                    with patch.object(service, "start_historical_data_collection", new_callable=AsyncMock) as mock_start:
                        resp_auto = await service.get_collection_status("BTC/USDT:USDT", "1h", mock_bg_tasks, True, mock_db)
                        assert resp_auto["status"] == "auto_fetch_started"

    @pytest.mark.asyncio
    async def test_start_bulk_collections(self, service, mock_db, mock_bg_tasks):
        # 1. Bitcoin full collection
        with patch("app.config.unified_config.unified_config.market.supported_timeframes", ["1h", "4h"]):
            resp = await service.start_bitcoin_full_data_collection(mock_bg_tasks, mock_db)
            assert resp["success"] is True
            assert mock_bg_tasks.add_task.call_count == 2
            
        # 2. Bulk historical collection
        mock_bg_tasks.reset_mock()
        with patch("app.config.unified_config.unified_config.market.supported_timeframes", ["1h"]):
            with patch("app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository") as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                mock_repo.get_data_count.return_value = 0
                
                resp = await service.start_bulk_historical_data_collection(mock_bg_tasks, mock_db)
                assert resp["success"] is True
                assert mock_bg_tasks.add_task.call_count > 0

    @pytest.mark.asyncio
    async def test_start_all_data_bulk_collection(self, service, mock_db, mock_bg_tasks):
        # 全データ一括収集の予約
        with patch("app.config.unified_config.unified_config.market.supported_timeframes", ["1h"]):
            with patch("app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository") as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                mock_repo.get_data_count.return_value = 0
                
                resp = await service.start_all_data_bulk_collection(mock_bg_tasks, mock_db)
                assert resp["success"] is True
                mock_bg_tasks.add_task.assert_called()

    @pytest.mark.asyncio
    async def test_historical_oi_collection(self, service, mock_db, mock_bg_tasks):
        # OI履歴収集
        with patch.object(service, "validate_symbol_and_timeframe", return_value="BTC"):
            resp = await service.start_historical_oi_collection("BTC", "1h", mock_bg_tasks, mock_db)
            assert resp["status"] == "started"
            mock_bg_tasks.add_task.assert_called()
            
        # 内部バックグラウンドタスクの導通
        with patch("app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService") as mock_oi_cls:
            mock_oi_cls.return_value.fetch_and_save_open_interest_data = AsyncMock(return_value={"success": True, "saved_count": 5})
            # db.query().count() 等のモック化は複雑なため、ここでは導通のみ確認
            await service._collect_historical_oi_background("BTC", "1h", mock_db)


    @pytest.mark.asyncio
    async def test_get_collection_status_exists(self, service, mock_db, mock_bg_tasks):
        # データがある場合のステータス確認
        from datetime import datetime
        with patch("app.config.unified_config.unified_config.market.supported_symbols", ["BTC/USDT:USDT"]):
            with patch("app.config.unified_config.unified_config.market.supported_timeframes", ["1h"]):
                with patch("app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository") as mock_repo_cls:
                    mock_repo = mock_repo_cls.return_value
                    mock_repo.get_data_count.return_value = 500
                    mock_repo.get_latest_timestamp.return_value = datetime(2023,1,1)
                    mock_repo.get_oldest_timestamp.return_value = datetime(2023,1,1)
                    
                    resp = await service.get_collection_status("BTC/USDT:USDT", "1h", mock_bg_tasks, False, mock_db)
                    assert resp["data"]["status"] == "data_exists"
                    assert resp["data"]["data_count"] == 500

    @pytest.mark.asyncio
    async def test_internal_background_tasks(self, service, mock_db):
        # _collect_historical_background の導通
        with patch.object(service.historical_service, "collect_historical_data_with_start_date", new_callable=AsyncMock) as mock_collect:
            mock_collect.return_value = 100
            await service._collect_historical_background("BTC", "1h", mock_db)
            mock_collect.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_all_data_background(self, service, mock_db):
        # _collect_all_data_background (OHLCV, FR, OI を順に呼ぶロジック)
        with patch.object(service.historical_service, "collect_historical_data", new_callable=AsyncMock) as mock_ohlcv:
            mock_ohlcv.return_value = 50
            
            # Funding/OI service をモック
            with patch("app.services.data_collection.bybit.funding_rate_service.BybitFundingRateService") as mock_fr_cls, \
                 patch("app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService") as mock_oi_cls:
                
                mock_fr_cls.return_value.fetch_and_save_funding_rate_data = AsyncMock(return_value={"success": True, "saved_count": 10})
                mock_oi_cls.return_value.fetch_and_save_open_interest_data = AsyncMock(return_value={"success": True, "saved_count": 10})
                
                await service._collect_all_data_background("BTC", "1h", mock_db)
                
                mock_ohlcv.assert_called_once()
                mock_fr_cls.return_value.fetch_and_save_funding_rate_data.assert_called_once()
                mock_oi_cls.return_value.fetch_and_save_open_interest_data.assert_called_once()
