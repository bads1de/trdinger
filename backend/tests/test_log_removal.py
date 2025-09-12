"""
ログ削除テスト

特定のINFOログメッセージが削除されていることを確認するテスト。
"""

import logging
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from database.repositories.ohlcv_repository import OHLCVRepository
from app.services.backtest.data.data_retrieval_service import DataRetrievalService
from app.services.backtest.data.data_integration_service import DataIntegrationService
from app.utils.data_processing.data_processor import DataProcessor
import pandas as pd


@pytest.fixture
def sample_ohlcv_data():
    """サンプルOHLCVデータ"""
    return [
        {
            'timestamp': datetime(2025, 6, 3, 0, 0, 0),
            'symbol': 'BTC/USDT:USDT',
            'timeframe': '1h',
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 100.0
        }
    ]


@pytest.fixture
def sample_dataframe():
    """サンプルDataFrame"""
    data = {
        'timestamp': [datetime(2025, 6, 3, 0, 0, 0)],
        'open': [50000.0],
        'high': [51000.0],
        'low': [49000.0],
        'close': [50500.0],
        'volume': [100.0],
        'open_interest': [1000000.0],
        'funding_rate': [0.0001]
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # timestampをdatetime型に
    return df


class TestLogRemoval:
    """ログ削除テスト"""

    def test_ohlcv_repository_no_query_log(self, sample_ohlcv_data, caplog):
        """OHLCVRepositoryのクエリログが出力されないことを確認"""
        with patch('database.repositories.ohlcv_repository.OHLCVRepository.get_filtered_data') as mock_get:
            mock_get.return_value = sample_ohlcv_data

            # Mock session
            mock_session = MagicMock()
            repo = OHLCVRepository(mock_session)

            with caplog.at_level(logging.INFO):
                repo.get_ohlcv_data(
                    symbol='BTC/USDT:USDT',
                    timeframe='1h',
                    start_time=datetime(2025, 6, 3, 0, 0, 0),
                    end_time=datetime(2025, 9, 3, 0, 0, 0)
                )

            # 削除されたログメッセージが含まれないことを確認
            log_messages = [record.message for record in caplog.records]
            assert "OHLCVRepository - Querying data:" not in "\n".join(log_messages)
            assert "OHLCVRepository - Found" not in "\n".join(log_messages)

    def test_data_retrieval_service_no_retrieved_log(self, sample_ohlcv_data, caplog):
        """DataRetrievalServiceの取得ログが出力されないことを確認"""
        with patch('database.repositories.ohlcv_repository.OHLCVRepository') as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.get_ohlcv_data.return_value = sample_ohlcv_data

            service = DataRetrievalService(ohlcv_repo=mock_repo)

            with caplog.at_level(logging.INFO):
                result = service.get_ohlcv_data(
                    symbol='BTC/USDT:USDT',
                    timeframe='1h',
                    start_date=datetime(2025, 6, 3, 0, 0, 0),
                    end_date=datetime(2025, 9, 3, 0, 0, 0)
                )

            # 削除されたログメッセージが含まれないことを確認
            log_messages = [record.message for record in caplog.records]
            assert "DataRetrievalService - Retrieved" not in "\n".join(log_messages)

    def test_data_integration_service_no_logs(self, sample_ohlcv_data, sample_dataframe, caplog):
        """DataIntegrationServiceのログが出力されないことを確認"""
        from app.services.backtest.data.data_retrieval_service import DataRetrievalService
        from app.services.backtest.data.data_conversion_service import DataConversionService

        with patch('app.services.backtest.data.data_retrieval_service.DataRetrievalService') as mock_retrieval:
            mock_retrieval_instance = mock_retrieval.return_value
            mock_retrieval_instance.get_ohlcv_data.return_value = sample_ohlcv_data

            with patch('app.services.backtest.data.data_conversion_service.DataConversionService') as mock_conversion:
                mock_conversion_instance = mock_conversion.return_value
                mock_conversion_instance.convert_ohlcv_to_dataframe.return_value = sample_dataframe

                service = DataIntegrationService(
                    retrieval_service=mock_retrieval_instance,
                    conversion_service=mock_conversion_instance
                )

                with caplog.at_level(logging.INFO):
                    result = service._get_base_ohlcv_dataframe(
                        symbol='BTC/USDT:USDT',
                        timeframe='1h',
                        start_date=datetime(2025, 6, 3, 0, 0, 0),
                        end_date=datetime(2025, 9, 3, 0, 0, 0)
                    )

                # 削除されたログメッセージが含まれないことを確認
                log_messages = [record.message for record in caplog.records]
                assert "DataIntegrationService - Retrieved" not in "\n".join(log_messages)
                assert "DataIntegrationService - Converted to DataFrame" not in "\n".join(log_messages)

    def test_data_processor_no_cleaning_logs(self, sample_dataframe, caplog):
        """DataProcessorのクリーニングログが出力されないことを確認"""
        processor = DataProcessor()

        with caplog.at_level(logging.INFO):
            result = processor.clean_and_validate_data(
                df=sample_dataframe,
                required_columns=['open', 'high', 'low', 'close', 'volume', 'open_interest', 'funding_rate']
            )

        # 削除されたログメッセージが含まれないことを確認
        log_messages = [record.message for record in caplog.records]
        assert "データクリーニングと検証を開始" not in "\n".join(log_messages)
        assert "カラム名を小文字に統一しました" not in "\n".join(log_messages)
        assert "拡張データの範囲クリップを開始" not in "\n".join(log_messages)

    def test_backtest_data_service_no_logs(self, sample_dataframe, caplog):
        """BacktestDataServiceのログが出力されないことを確認"""
        from app.services.backtest.backtest_data_service import BacktestDataService

        with patch('app.services.backtest.data.data_integration_service.DataIntegrationService') as mock_integration:
            mock_integration_instance = mock_integration.return_value
            mock_integration_instance.create_backtest_dataframe.return_value = sample_dataframe

            service = BacktestDataService()
            service._integration_service = mock_integration_instance

            with caplog.at_level(logging.INFO):
                result = service.get_data_for_backtest(
                    symbol='BTC/USDT:USDT',
                    timeframe='1h',
                    start_date=datetime(2025, 6, 3, 0, 0, 0),
                    end_date=datetime(2025, 9, 3, 0, 0, 0)
                )

            # 削除されたログメッセージが含まれないことを確認
            log_messages = [record.message for record in caplog.records]
            assert "BacktestDataService - DataFrame created with shape:" not in "\n".join(log_messages)
            assert "BacktestDataService - funding_rate stats:" not in "\n".join(log_messages)

    def test_backtest_executor_no_logs(self, sample_dataframe, caplog):
        """BacktestExecutorのログが出力されないことを確認"""
        from app.services.backtest.execution.backtest_executor import BacktestExecutor
        from app.services.backtest.backtest_data_service import BacktestDataService

        with patch('app.services.backtest.backtest_data_service.BacktestDataService') as mock_data_service:
            mock_data_service_instance = mock_data_service.return_value
            mock_data_service_instance.get_data_for_backtest.return_value = sample_dataframe

            executor = BacktestExecutor(mock_data_service_instance)

            with caplog.at_level(logging.INFO):
                result = executor._get_backtest_data(
                    symbol='BTC/USDT:USDT',
                    timeframe='1h',
                    start_date=datetime(2025, 6, 3, 0, 0, 0),
                    end_date=datetime(2025, 9, 3, 0, 0, 0)
                )

            # 削除されたログメッセージが含まれないことを確認
            log_messages = [record.message for record in caplog.records]
            assert "BacktestExecutor - Retrieved data shape:" not in "\n".join(log_messages)
            assert "BacktestExecutor - Data columns:" not in "\n".join(log_messages)
            assert "BacktestExecutor - Data head:" not in "\n".join(log_messages)