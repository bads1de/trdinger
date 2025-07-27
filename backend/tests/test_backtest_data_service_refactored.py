"""
リファクタリング後のBacktestDataServiceのテスト
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

import pandas as pd

from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.backtest.data.data_retrieval_service import DataRetrievalError
from app.services.backtest.data.data_conversion_service import DataConversionError
from app.services.backtest.data.data_integration_service import DataIntegrationError


class TestBacktestDataServiceRefactored:
    """リファクタリング後のBacktestDataServiceのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.mock_ohlcv_repo = Mock()
        self.mock_oi_repo = Mock()
        self.mock_fr_repo = Mock()
        self.mock_fear_greed_repo = Mock()
        
        self.service = BacktestDataService(
            ohlcv_repo=self.mock_ohlcv_repo,
            oi_repo=self.mock_oi_repo,
            fr_repo=self.mock_fr_repo,
            fear_greed_repo=self.mock_fear_greed_repo,
        )

    def test_initialization(self):
        """初期化のテスト"""
        service = BacktestDataService()
        assert service.ohlcv_repo is None
        assert service.oi_repo is None
        assert service.fr_repo is None
        assert service.fear_greed_repo is None
        assert service._retrieval_service is not None
        assert service._conversion_service is not None
        assert service._integration_service is not None

    def test_initialization_with_repositories(self):
        """リポジトリ付き初期化のテスト"""
        mock_ohlcv = Mock()
        mock_oi = Mock()
        mock_fr = Mock()
        mock_fg = Mock()
        
        service = BacktestDataService(
            ohlcv_repo=mock_ohlcv,
            oi_repo=mock_oi,
            fr_repo=mock_fr,
            fear_greed_repo=mock_fg,
        )
        
        assert service.ohlcv_repo == mock_ohlcv
        assert service.oi_repo == mock_oi
        assert service.fr_repo == mock_fr
        assert service.fear_greed_repo == mock_fg

    @patch('app.services.backtest.data.data_integration_service.DataIntegrationService')
    def test_get_data_for_backtest_success(self, mock_integration_service_class):
        """バックテスト用データ取得成功のテスト"""
        # モックの設定
        mock_integration_service = Mock()
        mock_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1100],
            'open_interest': [500, 510],
            'funding_rate': [0.001, 0.002]
        })
        mock_integration_service.create_backtest_dataframe.return_value = mock_df
        mock_integration_service_class.return_value = mock_integration_service

        service = BacktestDataService(
            ohlcv_repo=self.mock_ohlcv_repo,
            oi_repo=self.mock_oi_repo,
            fr_repo=self.mock_fr_repo,
        )
        service._integration_service = mock_integration_service

        result = service.get_data_for_backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        assert isinstance(result, pd.DataFrame)
        mock_integration_service.create_backtest_dataframe.assert_called_once_with(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            include_oi=True,
            include_fr=True,
            include_fear_greed=False,
        )

    @patch('app.services.backtest.data.data_integration_service.DataIntegrationService')
    def test_get_data_for_backtest_error(self, mock_integration_service_class):
        """バックテスト用データ取得エラーのテスト"""
        # モックの設定
        mock_integration_service = Mock()
        mock_integration_service.create_backtest_dataframe.side_effect = DataIntegrationError("Test error")
        mock_integration_service_class.return_value = mock_integration_service

        service = BacktestDataService(
            ohlcv_repo=self.mock_ohlcv_repo,
        )
        service._integration_service = mock_integration_service

        with pytest.raises(ValueError):
            service.get_data_for_backtest(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
            )

    @patch('app.services.backtest.data.data_integration_service.DataIntegrationService')
    def test_get_ml_training_data_success(self, mock_integration_service_class):
        """MLトレーニング用データ取得成功のテスト"""
        # モックの設定
        mock_integration_service = Mock()
        mock_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1100],
            'open_interest': [500, 510],
            'funding_rate': [0.001, 0.002],
            'fear_greed_value': [50, 55],
            'fear_greed_classification': ['Neutral', 'Greed']
        })
        mock_integration_service.create_ml_training_dataframe.return_value = mock_df
        mock_integration_service_class.return_value = mock_integration_service

        service = BacktestDataService(
            ohlcv_repo=self.mock_ohlcv_repo,
        )
        service._integration_service = mock_integration_service

        result = service.get_ml_training_data(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

        assert isinstance(result, pd.DataFrame)
        mock_integration_service.create_ml_training_dataframe.assert_called_once_with(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

    @patch('app.services.backtest.data.data_integration_service.DataIntegrationService')
    def test_get_ml_training_data_error(self, mock_integration_service_class):
        """MLトレーニング用データ取得エラーのテスト"""
        # モックの設定
        mock_integration_service = Mock()
        mock_integration_service.create_ml_training_dataframe.side_effect = DataIntegrationError("Test error")
        mock_integration_service_class.return_value = mock_integration_service

        service = BacktestDataService(
            ohlcv_repo=self.mock_ohlcv_repo,
        )
        service._integration_service = mock_integration_service

        with pytest.raises(ValueError):
            service.get_ml_training_data(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
            )

    @patch('app.services.backtest.data.data_integration_service.DataIntegrationService')
    def test_get_data_summary(self, mock_integration_service_class):
        """データ概要取得のテスト"""
        # モックの設定
        mock_integration_service = Mock()
        mock_summary = {"total_records": 100, "start_date": "2024-01-01"}
        mock_integration_service.get_data_summary.return_value = mock_summary
        mock_integration_service_class.return_value = mock_integration_service

        service = BacktestDataService()
        service._integration_service = mock_integration_service

        mock_df = pd.DataFrame()
        result = service.get_data_summary(mock_df)

        assert result == mock_summary
        mock_integration_service.get_data_summary.assert_called_once_with(mock_df)

    def test_backward_compatibility(self):
        """後方互換性のテスト"""
        service = BacktestDataService(
            ohlcv_repo=self.mock_ohlcv_repo,
            oi_repo=self.mock_oi_repo,
            fr_repo=self.mock_fr_repo,
            fear_greed_repo=self.mock_fear_greed_repo,
        )
        
        # 古いインターフェースが維持されていることを確認
        assert hasattr(service, 'ohlcv_repo')
        assert hasattr(service, 'oi_repo')
        assert hasattr(service, 'fr_repo')
        assert hasattr(service, 'fear_greed_repo')
        
        # 新しいサービスが初期化されていることを確認
        assert hasattr(service, '_retrieval_service')
        assert hasattr(service, '_conversion_service')
        assert hasattr(service, '_integration_service')
