import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.backtest.orchestration.backtest_orchestration_service import BacktestOrchestrationService

class TestBacktestOrchestrationService:
    @pytest.fixture
    def service(self):
        return BacktestOrchestrationService()

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_get_backtest_results(self, service, mock_db):
        with patch("app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_backtest_results.return_value = [{"id": 1}]
            mock_repo.count_backtest_results.return_value = 1
            
            result = await service.get_backtest_results(mock_db, limit=10, offset=0)
            
            assert result["success"] is True
            assert result["results"] == [{"id": 1}]
            assert result["total"] == 1
            mock_repo.get_backtest_results.assert_called_once_with(
                limit=10, offset=0, symbol=None, strategy_name=None
            )

    @pytest.mark.asyncio
    async def test_get_backtest_result_by_id_found(self, service, mock_db):
        with patch("app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_backtest_result_by_id.return_value = {"id": 1}
            
            # api_responseの結果をそのまま返すため、モックの戻り値を模倣
            # 実際には app.utils.response.api_response の挙動に依存する
            with patch("app.services.backtest.orchestration.backtest_orchestration_service.api_response", side_effect=lambda **kwargs: kwargs):
                result = await service.get_backtest_result_by_id(mock_db, 1)
                
                assert result["success"] is True
                assert result["data"] == {"id": 1}

    @pytest.mark.asyncio
    async def test_get_backtest_result_by_id_not_found(self, service, mock_db):
        with patch("app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_backtest_result_by_id.return_value = None
            
            with patch("app.services.backtest.orchestration.backtest_orchestration_service.api_response", side_effect=lambda **kwargs: kwargs):
                result = await service.get_backtest_result_by_id(mock_db, 999)
                
                assert result["success"] is False
                assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_delete_backtest_result_success(self, service, mock_db):
        with patch("app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository") as MockBacktestRepo, \
             patch("app.services.backtest.orchestration.backtest_orchestration_service.GeneratedStrategyRepository") as MockStrategyRepo:
            
            mock_backtest_repo = MockBacktestRepo.return_value
            mock_strategy_repo = MockStrategyRepo.return_value
            
            mock_backtest_repo.delete_backtest_result.return_value = True
            
            with patch("app.services.backtest.orchestration.backtest_orchestration_service.api_response", side_effect=lambda **kwargs: kwargs):
                result = await service.delete_backtest_result(mock_db, 1)
                
                assert result["success"] is True
                mock_strategy_repo.unlink_backtest_result.assert_called_once_with(1)
                mock_backtest_repo.delete_backtest_result.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_delete_all_backtest_results(self, service, mock_db):
        with patch("app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository") as MockBacktestRepo, \
             patch("app.services.backtest.orchestration.backtest_orchestration_service.GAExperimentRepository") as MockGARepo, \
             patch("app.services.backtest.orchestration.backtest_orchestration_service.GeneratedStrategyRepository") as MockStrategyRepo:
            
            MockBacktestRepo.return_value.delete_all_backtest_results.return_value = 10
            MockGARepo.return_value.delete_all_experiments.return_value = 5
            MockStrategyRepo.return_value.delete_all_strategies.return_value = 20
            
            with patch("app.services.backtest.orchestration.backtest_orchestration_service.api_response", side_effect=lambda **kwargs: kwargs):
                result = await service.delete_all_backtest_results(mock_db)
                
                assert result["success"] is True
                assert result["data"]["deleted_backtest_results"] == 10
                assert result["data"]["deleted_ga_experiments"] == 5
                assert result["data"]["deleted_generated_strategies"] == 20

    @pytest.mark.asyncio
    async def test_get_supported_strategies(self, service):
        with patch("app.services.backtest.orchestration.backtest_orchestration_service.BacktestService") as MockService:
            mock_bt_service = MockService.return_value
            mock_bt_service.get_supported_strategies.return_value = ["SMA", "RSI"]
            
            with patch("app.services.backtest.orchestration.backtest_orchestration_service.api_response", side_effect=lambda **kwargs: kwargs):
                result = await service.get_supported_strategies()
                
                assert result["success"] is True
                assert result["data"]["strategies"] == ["SMA", "RSI"]
                mock_bt_service.cleanup.assert_called_once()

