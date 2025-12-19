import pytest
import pandas as pd
import numpy as np
from app.utils.data_processing.validators.data_validator import (
    validate_ohlcv_data,
    validate_extended_data,
    validate_data_integrity
)
from app.utils.data_processing.validators.record_validator import RecordValidator

class TestValidatorsUnit:
    def test_validate_ohlcv_data_errors(self):
        # 1. カラム欠落
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv_data(pd.DataFrame({"open": [100]}))
            
        # 2. 非数値
        with pytest.raises(ValueError, match="not numeric"):
            validate_ohlcv_data(pd.DataFrame({
                "open": ["A"], "high": [105], "low": [95], "close": [100], "volume": [1000]
            }))
            
        # 3. 不整合 (Low > High)
        with pytest.raises(ValueError, match="OHLC values"):
            validate_ohlcv_data(pd.DataFrame({
                "open": [100], "high": [105], "low": [110], "close": [100], "volume": [1000]
            }))
            
        # 4. 負のボリューム
        with pytest.raises(ValueError, match="Volume contains negative values"):
            validate_ohlcv_data(pd.DataFrame({
                "open": [100], "high": [105], "low": [95], "close": [100], "volume": [-1]
            }))

    def test_validate_extended_data_errors(self):
        # funding_rate 範囲外
        with pytest.raises(ValueError, match="funding_rate values must be between -1 and 1"):
            validate_extended_data(pd.DataFrame({"funding_rate": [1.5]}))

    def test_validate_data_integrity_errors(self):
        # timestamp 型不正
        with pytest.raises(ValueError, match="timestamp column must be datetime type"):
            validate_data_integrity(pd.DataFrame({"timestamp": ["2023-01-01"]}))
            
        # ソート不正
        df = pd.DataFrame({"timestamp": pd.to_datetime(["2023-01-02", "2023-01-01"])})
        with pytest.raises(ValueError, match="sorted in ascending order"):
            validate_data_integrity(df)
            
        # 重複
        df_dup = pd.DataFrame({"timestamp": pd.to_datetime(["2023-01-01", "2023-01-01"])})
        with pytest.raises(ValueError, match="duplicate timestamps found"):
            validate_data_integrity(df_dup)

    def test_record_validator_logic(self):
        # 正常なレコード
        records = [{
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": "2023-01-01 00:00:00",
            "open": 100, "high": 105, "low": 95, "close": 100, "volume": 1000
        }]
        assert RecordValidator.validate_ohlcv_records_simple(records) is True
        
        # サニタイズ
        sanitized = RecordValidator.sanitize_ohlcv_data(records)
        assert sanitized[0]["symbol"] == "BTC/USDT"
        from datetime import datetime
        assert isinstance(sanitized[0]["timestamp"], (pd.Timestamp, datetime))

        
        # 異常なレコード (数値不正)
        bad_records = [{
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": "2023-01-01 00:00:00",
            "open": "invalid", "high": 105, "low": 95, "close": 100, "volume": 1000
        }]
        assert RecordValidator.validate_ohlcv_records_simple(bad_records) is False
        
        # 異常なレコード (必須フィールド欠落)
        missing_records = [{"open": 100}]
        assert RecordValidator.validate_ohlcv_records_simple(missing_records) is False
        
        # 無効な入力
        assert RecordValidator.validate_ohlcv_records_simple(None) is False
        assert RecordValidator.validate_ohlcv_records_simple("not a list") is False