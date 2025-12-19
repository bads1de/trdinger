import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session
from app.services.data_collection.orchestration.data_management_orchestration_service import DataManagementOrchestrationService

class TestDataManagementOrchestrationUnit:
    @pytest.fixture
    def service(self):
        return DataManagementOrchestrationService()

    @pytest.fixture
    def mock_session(self):
        return MagicMock(spec=Session)

    @pytest.mark.asyncio
    async def test_reset_all_data_success(self, service, mock_session):
        # 正常系: すべてのリセットが成功
        with patch("app.services.data_collection.orchestration.data_management_orchestration_service.OHLCVRepository") as mock_ohlcv_cls, \
             patch("app.services.data_collection.orchestration.data_management_orchestration_service.FundingRateRepository") as mock_fr_cls, \
             patch("app.services.data_collection.orchestration.data_management_orchestration_service.OpenInterestRepository") as mock_oi_cls:
            
            mock_ohlcv_cls.return_value.clear_all_ohlcv_data.return_value = 100
            mock_fr_cls.return_value.clear_all_funding_rate_data.return_value = 50
            mock_oi_cls.return_value.clear_all_open_interest_data.return_value = 30
            
            resp = await service.reset_all_data(db_session=mock_session)
            
            assert resp["success"] is True
            assert resp["data"]["deleted_counts"]["ohlcv"] == 100
            assert resp["data"]["total_deleted"] == 180

    @pytest.mark.asyncio
    async def test_reset_data_by_symbol_success(self, service, mock_session):
        # シンボル別リセット 正常系
        with patch("app.services.data_collection.orchestration.data_management_orchestration_service.OHLCVRepository") as mock_ohlcv_cls, \
             patch("app.services.data_collection.orchestration.data_management_orchestration_service.FundingRateRepository") as mock_fr_cls, \
             patch("app.services.data_collection.orchestration.data_management_orchestration_service.OpenInterestRepository") as mock_oi_cls:
            
            mock_ohlcv_cls.return_value.clear_ohlcv_data_by_symbol.return_value = 5
            mock_fr_cls.return_value.clear_funding_rate_data_by_symbol.return_value = 5
            mock_oi_cls.return_value.clear_open_interest_data_by_symbol.return_value = 5
            
            resp = await service.reset_data_by_symbol("BTC", mock_session)
            assert resp["success"] is True
            assert resp["data"]["symbol"] == "BTC"
            assert resp["data"]["total_deleted"] == 15

    @pytest.mark.asyncio
    async def test_reset_data_by_symbol_partial_fail(self, service, mock_session):
        # シンボル別リセット 部分的失敗 (FRのみ失敗)
        with patch("app.services.data_collection.orchestration.data_management_orchestration_service.OHLCVRepository") as mock_ohlcv_cls, \
             patch("app.services.data_collection.orchestration.data_management_orchestration_service.FundingRateRepository") as mock_fr_cls, \
             patch("app.services.data_collection.orchestration.data_management_orchestration_service.OpenInterestRepository") as mock_oi_cls:
            
            mock_ohlcv_cls.return_value.clear_ohlcv_data_by_symbol.return_value = 10
            mock_fr_cls.return_value.clear_funding_rate_data_by_symbol.side_effect = Exception("FR Fail")
            mock_oi_cls.return_value.clear_open_interest_data_by_symbol.return_value = 20
            
            resp = await service.reset_data_by_symbol("BTC", mock_session)
            assert resp["success"] is False
            assert "ファンディングレート削除エラー: FR Fail" in resp["data"]["errors"]

    @pytest.mark.asyncio
    async def test_get_data_status_success(self, service, mock_session):
        from datetime import datetime
        with patch("app.services.data_collection.orchestration.data_management_orchestration_service.OHLCVRepository") as mock_ohlcv_cls, \
             patch("app.services.data_collection.orchestration.data_management_orchestration_service.FundingRateRepository") as mock_fr_cls, \
             patch("app.services.data_collection.orchestration.data_management_orchestration_service.OpenInterestRepository") as mock_oi_cls:
            
            mock_ohlcv_cls.return_value.get_data_count.return_value = 10
            mock_ohlcv_cls.return_value.get_latest_timestamp.return_value = datetime(2023,1,1)
            mock_fr_cls.return_value.get_latest_funding_timestamp.return_value = datetime(2023,1,1)
            mock_oi_cls.return_value.get_latest_open_interest_timestamp.return_value = datetime(2023,1,1)
            mock_session.query.return_value.count.return_value = 100
            
            resp = await service.get_data_status(mock_session)
            assert resp["success"] is True
            assert resp["data"]["total_records"] == 300

    @pytest.mark.asyncio
    async def test_reset_individual_types(self, service, mock_session):
        # 単体リセットメソッド群のテスト
        with patch("app.services.data_collection.orchestration.data_management_orchestration_service.OHLCVRepository") as mock_repo:
            mock_repo.return_value.clear_all_ohlcv_data.return_value = 1
            await service.reset_ohlcv_data(mock_session)
        with patch("app.services.data_collection.orchestration.data_management_orchestration_service.FundingRateRepository") as mock_repo:
            mock_repo.return_value.clear_all_funding_rate_data.return_value = 1
            await service.reset_funding_rate_data(mock_session)
        with patch("app.services.data_collection.orchestration.data_management_orchestration_service.OpenInterestRepository") as mock_repo:
            mock_repo.return_value.clear_all_open_interest_data.return_value = 1
            await service.reset_open_interest_data(mock_session)

    def test_get_db_session_logic(self, service, mock_session):
        # セッション取得ロジック
        with service._get_db_session(mock_session) as s:
            assert s == mock_session
        with patch("app.services.data_collection.orchestration.data_management_orchestration_service.SessionLocal") as mock_local:
            mock_local.return_value = MagicMock()
            with service._get_db_session(None) as s:
                assert s is not None

    @pytest.mark.asyncio
    async def test_fatal_errors(self, service, mock_session):
        # 致命的な例外パス
        with patch.object(service, "_get_db_session", side_effect=Exception("Fatal")):
            res1 = await service.reset_all_data(mock_session)
            assert "エラーが発生しました" in res1["message"]
            res2 = await service.get_data_status(mock_session)
            assert "エラーが発生しました" in res2["message"]