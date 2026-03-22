import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock
from app.services.backtest.data.data_conversion_service import DataConversionService

class TestDataConversionService:
    @pytest.fixture
    def service(self):
        return DataConversionService()

    def test_convert_ohlcv_to_dataframe_success(self, service):
        # データベースモデルを模倣したオブジェクト
        mock_data = []
        for i in range(3):
            m = MagicMock()
            m.timestamp = datetime(2023, 1, 1, i)
            m.open = 100.0 + i
            m.high = 105.0 + i
            m.low = 95.0 + i
            m.close = 101.0 + i
            m.volume = 1000.0 * i
            mock_data.append(m)
            
        df = service.convert_ohlcv_to_dataframe(mock_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "open" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df["open"].iloc[0] == 100.0
        assert df["volume"].dtype == "float64"

    def test_convert_ohlcv_to_dataframe_empty(self, service):
        df = service.convert_ohlcv_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_optimize_ohlcv_dtypes(self, service):
        df = pd.DataFrame({
            "timestamp": ["2023-01-01"],
            "open": ["100"],
            "volume": [1000]
        })
        
        optimized = service._optimize_ohlcv_dtypes(df)
        
        assert optimized["open"].dtype == "float64"
        assert pd.api.types.is_datetime64_any_dtype(optimized["timestamp"])
