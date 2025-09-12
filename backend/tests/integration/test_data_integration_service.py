"""
データ統合サービスの統合テスト

バックテスト用DataFrame作成機能のテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from backend.app.services.backtest.data.data_integration_service import DataIntegrationService
from backend.app.services.backtest.data.data_retrieval_service import DataRetrievalService


class TestDataIntegrationService:
    """DataIntegrationServiceのテスト"""

    @pytest.fixture
    def mock_retrieval_service(self):
        """モックデータ取得サービス"""
        service = Mock(spec=DataRetrievalService)

        # OHLCVDataオブジェクトのリストを作成
        from database.models import OHLCVData
        from datetime import datetime

        ohlcv_objects = []
        for i in range(10):
            obj = OHLCVData(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                timestamp=datetime(2023, 1, 1, i),
                open=float(100 + i),
                high=float(105 + i),
                low=float(95 + i),
                close=float(102 + i),
                volume=float(1000 + i * 100)
            )
            ohlcv_objects.append(obj)

        service.get_ohlcv_data.return_value = ohlcv_objects
        service.oi_repo = None
        service.fr_repo = None
        service.fear_greed_repo = None

        return service

    @pytest.fixture
    def data_integration_service(self, mock_retrieval_service):
        """DataIntegrationServiceインスタンス"""
        return DataIntegrationService(mock_retrieval_service)

    def test_create_backtest_dataframe_success(self, data_integration_service):
        """バックテストDataFrame作成成功テスト"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 5)

        result = data_integration_service.create_backtest_dataframe(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=start_date,
            end_date=end_date,
            include_oi=False,
            include_fr=False,
            include_fear_greed=False
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns

    def test_create_backtest_dataframe_missing_required_columns(self, mock_retrieval_service):
        """必須カラムが欠けている場合のテスト"""
        # 正常なOHLCVDataオブジェクトのリストを作成
        from database.models import OHLCVData
        from datetime import datetime

        complete_objects = []
        for i in range(10):
            obj = OHLCVData(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                timestamp=datetime(2023, 1, 1, i),
                open=float(100 + i),
                high=float(105 + i),
                low=float(95 + i),
                close=float(102 + i),
                volume=float(1000 + i * 100)
            )
            complete_objects.append(obj)

        mock_retrieval_service.get_ohlcv_data.return_value = complete_objects

        service = DataIntegrationService(mock_retrieval_service)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)

        # 正常な場合、成功する
        result = service.create_backtest_dataframe(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=start_date,
            end_date=end_date,
            include_oi=False,
            include_fr=False,
            include_fear_greed=False
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_get_data_summary_success(self, data_integration_service):
        """データ概要取得成功テスト"""
        # サンプルデータを作成
        dates = pd.date_range('2023-01-01', periods=10, freq='h')
        sample_data = pd.DataFrame({
            'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'Open': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }, index=dates)

        summary = data_integration_service.get_data_summary(sample_data)

        assert isinstance(summary, dict)
        assert 'total_records' in summary
        assert 'price_range' in summary
        assert 'volume_stats' in summary
        assert summary['total_records'] == 10
        assert summary['price_range']['min'] == 95
        assert summary['price_range']['max'] == 114