"""
リファクタリング後のBacktestServiceのテスト
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from app.services.backtest.backtest_service import BacktestService
from app.services.backtest.validation.backtest_config_validator import (
    BacktestConfigValidationError
)
from app.services.backtest.factories.strategy_class_factory import (
    StrategyClassCreationError
)
from app.services.backtest.execution.backtest_executor import (
    BacktestExecutionError
)
from app.services.backtest.conversion.backtest_result_converter import (
    BacktestResultConversionError
)


class TestBacktestServiceRefactored:
    """リファクタリング後のBacktestServiceのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.mock_data_service = Mock()
        self.service = BacktestService(data_service=self.mock_data_service)

    def test_initialization(self):
        """初期化のテスト"""
        service = BacktestService()
        assert service.data_service is None
        assert service._validator is not None
        assert service._strategy_factory is not None
        assert service._result_converter is not None
        assert service._executor is None

    def test_initialization_with_data_service(self):
        """データサービス付き初期化のテスト"""
        mock_data_service = Mock()
        service = BacktestService(data_service=mock_data_service)
        assert service.data_service == mock_data_service

    @patch('app.services.backtest_service.BacktestConfigValidator')
    def test_config_validation_error(self, mock_validator_class):
        """設定検証エラーのテスト"""
        # モックの設定
        mock_validator = Mock()
        mock_validator.validate_config.side_effect = BacktestConfigValidationError("Invalid config")
        mock_validator_class.return_value = mock_validator

        service = BacktestService(data_service=self.mock_data_service)
        
        config = {
            "strategy_name": "test",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000,
            "commission_rate": 0.001,
            "strategy_config": {}
        }

        with pytest.raises(BacktestConfigValidationError):
            service.run_backtest(config)

    @patch('app.services.backtest_service.StrategyClassFactory')
    def test_strategy_creation_error(self, mock_factory_class):
        """戦略クラス生成エラーのテスト"""
        # モックの設定
        mock_factory = Mock()
        mock_factory.create_strategy_class.side_effect = StrategyClassCreationError("Strategy creation failed")
        mock_factory_class.return_value = mock_factory

        service = BacktestService(data_service=self.mock_data_service)
        
        config = {
            "strategy_name": "test",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000,
            "commission_rate": 0.001,
            "strategy_config": {"strategy_gene": {}}
        }

        with pytest.raises(StrategyClassCreationError):
            service.run_backtest(config)

    def test_normalize_date_datetime(self):
        """日付正規化のテスト（datetime）"""
        service = BacktestService(data_service=self.mock_data_service)
        test_date = datetime(2024, 1, 1)
        result = service._normalize_date(test_date)
        assert result == test_date

    def test_normalize_date_string(self):
        """日付正規化のテスト（文字列）"""
        service = BacktestService(data_service=self.mock_data_service)
        test_date_str = "2024-01-01T00:00:00"
        result = service._normalize_date(test_date_str)
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_normalize_date_invalid(self):
        """日付正規化のテスト（無効な形式）"""
        service = BacktestService(data_service=self.mock_data_service)
        with pytest.raises(ValueError):
            service._normalize_date(123)

    @patch('app.services.backtest_service.get_db')
    @patch('app.services.backtest_service.OHLCVRepository')
    @patch('app.services.backtest_service.OpenInterestRepository')
    @patch('app.services.backtest_service.FundingRateRepository')
    @patch('app.services.backtest_service.BacktestDataService')
    def test_ensure_data_service_initialized(
        self, mock_data_service_class, mock_fr_repo, mock_oi_repo, mock_ohlcv_repo, mock_get_db
    ):
        """データサービス初期化のテスト"""
        # モックの設定
        mock_db = Mock()
        mock_get_db.return_value = iter([mock_db])
        
        service = BacktestService()  # data_serviceなしで初期化
        service._ensure_data_service_initialized()
        
        # データサービスが初期化されたことを確認
        assert service.data_service is not None
        mock_data_service_class.assert_called_once()

    def test_ensure_executor_initialized(self):
        """実行エンジン初期化のテスト"""
        service = BacktestService(data_service=self.mock_data_service)
        service._ensure_executor_initialized()
        
        # 実行エンジンが初期化されたことを確認
        assert service._executor is not None

    @patch('app.services.backtest_service.BacktestExecutor')
    def test_get_supported_strategies(self, mock_executor_class):
        """サポート戦略取得のテスト"""
        # モックの設定
        mock_executor = Mock()
        mock_executor.get_supported_strategies.return_value = {"test": "strategy"}
        mock_executor_class.return_value = mock_executor

        service = BacktestService(data_service=self.mock_data_service)
        result = service.get_supported_strategies()
        
        assert result == {"test": "strategy"}
        mock_executor.get_supported_strategies.assert_called_once()

    def test_execute_and_save_backtest_success(self):
        """バックテスト実行・保存成功のテスト"""
        # モックリクエスト
        mock_request = Mock()
        mock_request.strategy_name = "test_strategy"
        mock_request.symbol = "BTC/USDT"
        mock_request.timeframe = "1h"
        mock_request.start_date = "2024-01-01"
        mock_request.end_date = "2024-01-02"
        mock_request.initial_capital = 10000
        mock_request.commission_rate = 0.001
        mock_request.strategy_config.dict.return_value = {"test": "config"}

        # モックセッション
        mock_session = Mock()

        # run_backtestをモック
        with patch.object(self.service, 'run_backtest') as mock_run_backtest:
            mock_run_backtest.return_value = {"test": "result"}
            
            # BacktestResultRepositoryをモック
            with patch('app.services.backtest_service.BacktestResultRepository') as mock_repo_class:
                mock_repo = Mock()
                mock_repo.save_backtest_result.return_value = {"saved": "result"}
                mock_repo_class.return_value = mock_repo

                result = self.service.execute_and_save_backtest(mock_request, mock_session)

                assert result == {"success": True, "result": {"saved": "result"}}
                mock_run_backtest.assert_called_once()
                mock_repo.save_backtest_result.assert_called_once_with({"test": "result"})

    def test_execute_and_save_backtest_error(self):
        """バックテスト実行・保存エラーのテスト"""
        mock_request = Mock()
        mock_session = Mock()

        # run_backtestでエラーを発生させる
        with patch.object(self.service, 'run_backtest') as mock_run_backtest:
            mock_run_backtest.side_effect = Exception("Test error")

            with pytest.raises(Exception):
                self.service.execute_and_save_backtest(mock_request, mock_session)
