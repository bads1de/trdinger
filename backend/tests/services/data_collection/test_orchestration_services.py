
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from app.services.data_collection.orchestration.funding_rate_orchestration_service import FundingRateOrchestrationService
from app.services.data_collection.orchestration.open_interest_orchestration_service import OpenInterestOrchestrationService
from app.services.data_collection.orchestration.long_short_ratio_orchestration_service import LongShortRatioOrchestrationService

class TestOrchestrationServices:
    
    @pytest.fixture
    def mock_db_session(self):
        return MagicMock()

    @pytest.fixture
    def mock_funding_service(self):
        service = MagicMock()
        service.fetch_all_funding_rate_history = AsyncMock(return_value=[{"timestamp": 1000, "rate": 0.01}])
        service.fetch_funding_rate_history = AsyncMock(return_value=[{"timestamp": 1000, "rate": 0.01}])
        return service

    @pytest.fixture
    def mock_open_interest_service(self):
        service = MagicMock()
        service.fetch_and_save_open_interest_data = AsyncMock(return_value={"success": True, "saved_count": 10})
        return service

    @pytest.fixture
    def mock_long_short_service(self):
        service = MagicMock()
        # 履歴収集のモック
        service.collect_historical_long_short_ratio_data = AsyncMock(return_value=50)
        # 差分更新のモック
        service.fetch_incremental_long_short_ratio_data = AsyncMock(return_value={"success": True, "saved_count": 5})
        return service

    @pytest.mark.asyncio
    async def test_funding_rate_collection(self, mock_db_session, mock_funding_service):
        """FundingRateOrchestrationServiceのテスト"""
        
        # リポジトリのモック
        with patch("app.services.data_collection.orchestration.funding_rate_orchestration_service.FundingRateRepository") as MockRepo:
            mock_repo_instance = MockRepo.return_value
            mock_repo_instance.insert_funding_rate_data.return_value = 10

            service = FundingRateOrchestrationService(bybit_service=mock_funding_service)
            
            result = await service.collect_funding_rate_data(
                symbol="BTCUSDT",
                limit=100,
                fetch_all=False,
                db_session=mock_db_session
            )
            
            assert result["success"] is True
            assert result["data"]["count"] == 10
            mock_funding_service.fetch_funding_rate_history.assert_called_once()
            mock_repo_instance.insert_funding_rate_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_open_interest_collection(self, mock_db_session, mock_open_interest_service):
        """OpenInterestOrchestrationServiceのテスト"""
        
        # リポジトリのモック
        with patch("app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository") as MockRepo:
            
            service = OpenInterestOrchestrationService(bybit_service=mock_open_interest_service)
            
            result = await service.collect_open_interest_data(
                symbol="BTCUSDT",
                limit=100,
                fetch_all=False,
                db_session=mock_db_session
            )
            
            # api_responseの構造を確認 (successキーがあるかなど)
            assert result["success"] is True
            assert "saved_count" in result["data"]
            mock_open_interest_service.fetch_and_save_open_interest_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_long_short_ratio_collection(self, mock_db_session, mock_long_short_service):
        """LongShortRatioOrchestrationServiceのテスト"""
        
        with patch("app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository") as MockRepo:
            
            service = LongShortRatioOrchestrationService(bybit_service=mock_long_short_service)
            
            # ケース1: fetch_all=True (履歴収集)
            result_hist = await service.collect_long_short_ratio_data(
                symbol="BTCUSDT",
                period="5min",
                fetch_all=True,
                db_session=mock_db_session
            )
            
            assert result_hist["success"] is True
            assert result_hist["data"]["count"] == 50
            mock_long_short_service.collect_historical_long_short_ratio_data.assert_called_once()
            
            # ケース2: fetch_all=False (差分更新)
            result_incr = await service.collect_long_short_ratio_data(
                symbol="BTCUSDT",
                period="5min",
                fetch_all=False,
                db_session=mock_db_session
            )
            
            assert result_incr["success"] is True
            assert result_incr["data"]["count"] == 5
            mock_long_short_service.fetch_incremental_long_short_ratio_data.assert_called_once()

    def test_parse_datetime(self):
        """_parse_datetimeの共通化可能性を確認"""
        service = FundingRateOrchestrationService(bybit_service=MagicMock())
        
        # 正常系
        dt = service._parse_datetime("2023-01-01T00:00:00")
        assert isinstance(dt, datetime)
        assert dt.year == 2023
        
        # 異常系
        assert service._parse_datetime(None) is None
        assert service._parse_datetime("invalid-date") is None




