import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from app.utils.data_conversion import (
    parse_timestamp_safe,
    OHLCVDataConverter,
    FundingRateDataConverter,
    OpenInterestDataConverter
)

class TestDataConversionUnit:
    def test_parse_timestamp_safe(self):
        # 1. 正常系: datetime
        now = datetime.now(timezone.utc)
        assert parse_timestamp_safe(now) == now
        
        # 2. 正常系: str (ISO)
        ts_str = "2023-01-01T00:00:00Z"
        expected = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert parse_timestamp_safe(ts_str) == expected
        
        # 3. 正常系: int/float (ミリ秒)
        ts_ms = 1672531200000 # 2023-01-01 00:00:00 UTC
        assert parse_timestamp_safe(ts_ms) == expected
        assert parse_timestamp_safe(float(ts_ms)) == expected
        
        # 4. 異常系: None
        assert parse_timestamp_safe(None) is None
        
        # 5. 異常系: 不明な型
        assert parse_timestamp_safe([123]) is None
        
        # 6. 異常系: 無効な文字列/数値
        assert parse_timestamp_safe("invalid date") is None
        assert parse_timestamp_safe(-9999999999999) is None

    def test_ohlcv_data_converter(self):
        # CCXT -> DB
        ccxt_data = [
            [1672531200000, 100.0, 105.0, 95.0, 101.0, 1000.0],
            [1672534800000, 101.0, 106.0, 100.0, 102.0, 1100.0]
        ]
        db_records = OHLCVDataConverter.ccxt_to_db_format(ccxt_data, "BTC/USDT", "1h")
        
        assert len(db_records) == 2
        assert db_records[0]["symbol"] == "BTC/USDT"
        assert db_records[0]["open"] == 100.0
        assert isinstance(db_records[0]["timestamp"], datetime)
        
        # 異常系: 要素数不足
        bad_ccxt = [[1672531200000, 100.0]]
        assert len(OHLCVDataConverter.ccxt_to_db_format(bad_ccxt, "BTC", "1h")) == 0
        
        # DB -> API
        # ダミーのレコードオブジェクト
        class MockRecord:
            def __init__(self, **kwargs):
                for k, v in kwargs.items(): setattr(self, k, v)
        
        db_objs = [
            MockRecord(timestamp=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), open=100, high=105, low=95, close=101, volume=1000)
        ]
        api_data = OHLCVDataConverter.db_to_api_format(db_objs)
        assert len(api_data) == 1
        assert api_data[0][0] == 1672531200000
        assert api_data[0][1] == 100.0

    def test_funding_rate_converter(self):
        ccxt_funding = [
            {
                "datetime": "2023-01-01T00:00:00Z",
                "fundingRate": 0.0001,
                "nextFundingDatetime": "2023-01-01T08:00:00Z"
            }
        ]
        db_records = FundingRateDataConverter.ccxt_to_db_format(ccxt_funding, "BTC/USDT")
        assert len(db_records) == 1
        assert db_records[0]["funding_rate"] == 0.0001
        assert "next_funding_timestamp" in db_records[0]

    def test_open_interest_converter(self):
        # 取引所によるフィールド名の違いを網羅
        ccxt_oi = [
            {"timestamp": 1672531200000, "openInterestAmount": 500.0},
            {"timestamp": 1672534800000, "openInterest": 600.0},
            {"timestamp": 1672538400000, "openInterestValue": 700.0},
            {"timestamp": 1672542000000, "badField": 800.0} # スキップされるはず
        ]
        db_records = OpenInterestDataConverter.ccxt_to_db_format(ccxt_oi, "BTC/USDT")
        assert len(db_records) == 3 # 3つ成功
        assert db_records[0]["open_interest_value"] == 500.0
        assert db_records[1]["open_interest_value"] == 600.0
        assert db_records[2]["open_interest_value"] == 700.0
