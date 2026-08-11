from unittest.mock import MagicMock, patch

import pytest

from app.services.backtest.execution.backtest_executor import BacktestExecutionError
from app.services.backtest.services.backtest_service import BacktestService


class TestBacktestServiceEnhancement:
    @pytest.fixture
    def service(self):
        return BacktestService()

    def test_ensure_data_service_initialized_success(self, service):
        # get_dbをモック
        mock_db_gen = MagicMock()
        mock_session = MagicMock()
        mock_db_gen.__next__.return_value = mock_session

        with patch(
            "app.services.backtest.services.backtest_service.get_db",
            return_value=mock_db_gen,
        ):
            service.ensure_data_service_initialized()

            assert service.data_service is not None
            assert service._db_session == mock_session

    def test_ensure_data_service_initialized_failure(self, service):
        with patch(
            "app.services.backtest.services.backtest_service.get_db"
        ) as mock_get_db:
            mock_get_db.return_value.__next__.side_effect = Exception(
                "DB connection error"
            )
            with pytest.raises(BacktestExecutionError) as excinfo:
                service.ensure_data_service_initialized()
            assert "初期化に失敗しました" in str(excinfo.value)

    def test_ensure_orchestrator_initialized_success(self, service):
        service.data_service = MagicMock()
        service._ensure_orchestrator_initialized()
        assert service._orchestrator is not None

    def test_ensure_orchestrator_initialized_no_data_service(self, service):
        with pytest.raises(BacktestExecutionError) as excinfo:
            service._ensure_orchestrator_initialized()
        assert "データサービスが初期化されていません" in str(excinfo.value)

    def test_run_backtest_full_flow(self, service):
        # 内部メソッドをモック
        service.ensure_data_service_initialized = MagicMock()
        service._ensure_orchestrator_initialized = MagicMock()

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = {"result": "ok"}
        service._orchestrator = mock_orchestrator

        result = service.run_backtest({"symbol": "BTC"})

        assert result == {"result": "ok"}
        service.ensure_data_service_initialized.assert_called_once()
        service._ensure_orchestrator_initialized.assert_called_once()

    def test_cleanup(self, service):
        mock_session = MagicMock()
        service._db_session = mock_session

        service.cleanup()

        mock_session.close.assert_called_once()
        assert service._db_session is None

    def test_get_supported_strategies(self, service):
        service.ensure_data_service_initialized = MagicMock()
        service._ensure_orchestrator_initialized = MagicMock()

        mock_orchestrator = MagicMock()
        mock_orchestrator.get_supported_strategies.return_value = {"strat": {}}
        service._orchestrator = mock_orchestrator

        result = service.get_supported_strategies()
        assert "strat" in result
