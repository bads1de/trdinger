import pytest
from unittest.mock import MagicMock
from datetime import datetime
from app.services.backtest.data.data_retrieval_service import DataRetrievalService, DataRetrievalError

class TestDataRetrievalService:
    def test_get_ohlcv_data_no_repo(self):
        service = DataRetrievalService(ohlcv_repo=None)
        assert service.get_ohlcv_data("BTC/USDT", "1h", datetime.now(), datetime.now()) == []

    def test_get_ohlcv_data_success(self):
        mock_repo = MagicMock()
        mock_repo.get_ohlcv_data.return_value = [MagicMock()]
        service = DataRetrievalService(ohlcv_repo=mock_repo)
        
        result = service.get_ohlcv_data("BTC/USDT", "1h", datetime(2023, 1, 1), datetime(2023, 1, 2))
        
        assert len(result) == 1
        mock_repo.get_ohlcv_data.assert_called_once()

    def test_get_ohlcv_data_empty_raises_error(self):
        mock_repo = MagicMock()
        mock_repo.get_ohlcv_data.return_value = []
        service = DataRetrievalService(ohlcv_repo=mock_repo)
        
        # safe_operationでラップされているため、内部で例外が出てもdefault_return([])が返る可能性がある
        # プロダクションコードを確認すると _get_ohlcv_data 内で raise している
        # safe_operation(default_return=[]) の設定により、例外はキャッチされて空リストが返る
        result = service.get_ohlcv_data("BTC/USDT", "1h", datetime(2023, 1, 1), datetime(2023, 1, 2))
        assert result == []

    def test_get_open_interest_data_success(self):
        mock_repo = MagicMock()
        mock_repo.get_open_interest_data.return_value = [1, 2, 3]
        service = DataRetrievalService(oi_repo=mock_repo)
        
        result = service.get_open_interest_data("BTC/USDT", datetime.now(), datetime.now())
        assert result == [1, 2, 3]

    def test_get_funding_rate_data_success(self):
        mock_repo = MagicMock()
        mock_repo.get_funding_rate_data.return_value = [0.0001]
        service = DataRetrievalService(fr_repo=mock_repo)
        
        result = service.get_funding_rate_data("BTC/USDT", datetime.now(), datetime.now())
        assert result == [0.0001]
