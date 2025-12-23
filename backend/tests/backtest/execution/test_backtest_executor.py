import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime
from app.services.backtest.execution.backtest_executor import BacktestExecutor, BacktestExecutionError

class TestBacktestExecutor:
    @pytest.fixture
    def data_service(self):
        return MagicMock()

    @pytest.fixture
    def executor(self, data_service):
        return BacktestExecutor(data_service)

    @pytest.fixture
    def sample_data(self):
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [10, 20]
        }, index=pd.to_datetime(["2023-01-01 10:00", "2023-01-01 11:00"]))
        return df

    def test_get_backtest_data_success(self, executor, data_service, sample_data):
        data_service.get_data_for_backtest.return_value = sample_data
        
        result = executor._get_backtest_data("BTC/USDT", "1h", datetime(2023, 1, 1), datetime(2023, 1, 2))
        
        assert not result.empty
        assert "open" in result.columns
        data_service.get_data_for_backtest.assert_called_once()

    def test_get_backtest_data_empty(self, executor, data_service):
        data_service.get_data_for_backtest.return_value = pd.DataFrame()
        
        with pytest.raises(BacktestExecutionError) as excinfo:
            executor._get_backtest_data("BTC/USDT", "1h", datetime(2023, 1, 1), datetime(2023, 1, 2))
        assert "データが見つかりませんでした" in str(excinfo.value)

    def test_create_backtest_instance(self, executor, sample_data):
        mock_strategy = MagicMock()
        
        # FractionalBacktestの初期化をパッチ
        with patch("app.services.backtest.execution.backtest_executor.FractionalBacktest") as MockBT:
            bt = executor._create_backtest_instance(sample_data, mock_strategy, 10000, 0.001, 0.0005, 2.0, "BTC/USDT")
            
            assert bt is not None
            # カラム名が大文字になっているか確認
            args, kwargs = MockBT.call_args
            df_passed = args[0]
            assert "Open" in df_passed.columns
            assert "Close" in df_passed.columns
            assert kwargs["cash"] == 10000
            assert kwargs["commission"] == 0.0015
            assert kwargs["margin"] == 0.5

    def test_run_backtest(self, executor):
        mock_bt = MagicMock()
        mock_stats = MagicMock()
        mock_bt.run.return_value = mock_stats
        
        params = {"p1": 10}
        result = executor._run_backtest(mock_bt, params)
        
        assert result == mock_stats
        mock_bt.run.assert_called_once_with(p1=10)

    def test_execute_backtest_full_flow(self, executor, data_service, sample_data):
        mock_strategy = MagicMock()
        mock_stats = MagicMock()
        data_service.get_data_for_backtest.return_value = sample_data
        
        with patch.object(executor, '_create_backtest_instance') as mock_create, \
             patch.object(executor, '_run_backtest', return_value=mock_stats) as mock_run:
            
            result = executor.execute_backtest(
                mock_strategy, {"p1": 1}, "BTC/USDT", "1h", 
                datetime(2023, 1, 1), datetime(2023, 1, 2), 10000, 0.001,
                0.0, 1.0
            )
            
            assert result == mock_stats
            mock_create.assert_called_once()
            mock_run.assert_called_once()

    def test_execute_backtest_preloaded(self, executor, sample_data):
        mock_strategy = MagicMock()
        
        with patch.object(executor, '_create_backtest_instance') as mock_create, \
             patch.object(executor, '_run_backtest') as mock_run:
            
            executor.execute_backtest(
                mock_strategy, {}, "BTC/USDT", "1h", 
                datetime(2023, 1, 1), datetime(2023, 1, 2), 10000, 0.001,
                0.0, 1.0,
                preloaded_data=sample_data
            )
            
            # 内部のデータ取得が呼ばれないことを確認（data_serviceが呼ばれない）
            executor.data_service.get_data_for_backtest.assert_not_called()
            mock_create.assert_called_once()

    def test_get_supported_strategies(self, executor):
        strategies = executor.get_supported_strategies()
        assert "auto_strategy" in strategies
        assert strategies["auto_strategy"]["name"] == "オートストラテジー"
