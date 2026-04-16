from datetime import datetime
from unittest.mock import MagicMock

import pytest

from app.services.backtest.data.data_retrieval_service import (
    DataRetrievalError,
    DataRetrievalService,
)


class TestDataRetrievalService:
    def test_get_ohlcv_data_no_repo(self):
        service = DataRetrievalService(ohlcv_repo=None)
        assert (
            service.get_ohlcv_data("BTC/USDT", "1h", datetime.now(), datetime.now())
            == []
        )

    def test_get_ohlcv_data_success(self):
        mock_repo = MagicMock()
        mock_repo.get_ohlcv_data.return_value = [MagicMock()]
        service = DataRetrievalService(ohlcv_repo=mock_repo)

        result = service.get_ohlcv_data(
            "BTC/USDT", "1h", datetime(2023, 1, 1), datetime(2023, 1, 2)
        )

        assert len(result) == 1
        mock_repo.get_ohlcv_data.assert_called_once()

    def test_get_ohlcv_data_empty_raises_error(self):
        mock_repo = MagicMock()
        mock_repo.get_ohlcv_data.return_value = []
        service = DataRetrievalService(ohlcv_repo=mock_repo)

        # raise_on_empty=True の場合、空データは DataRetrievalError を送出する
        with pytest.raises(DataRetrievalError):
            service.get_ohlcv_data(
                "BTC/USDT", "1h", datetime(2023, 1, 1), datetime(2023, 1, 2)
            )

    def test_get_open_interest_data_success(self):
        mock_repo = MagicMock()
        mock_repo.get_open_interest_data.return_value = [1, 2, 3]
        service = DataRetrievalService(oi_repo=mock_repo)

        result = service.get_open_interest_data(
            "BTC/USDT", datetime.now(), datetime.now()
        )
        assert result == [1, 2, 3]

    def test_get_funding_rate_data_success(self):
        mock_repo = MagicMock()
        mock_repo.get_funding_rate_data.return_value = [0.0001]
        service = DataRetrievalService(fr_repo=mock_repo)

        result = service.get_funding_rate_data(
            "BTC/USDT", datetime.now(), datetime.now()
        )
        assert result == [0.0001]
