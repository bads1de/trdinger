"""
RecordValidator のユニットテスト

レコード形式（辞書リスト）バリデータの全メソッドをテストします:
- validate_ohlcv_records_simple
- sanitize_ohlcv_data
- DataValidator エイリアス
"""

from datetime import datetime

import pandas as pd
import pytest

from app.utils.data_processing.record_validator import (
    DataValidator,
    RecordValidator,
)


@pytest.fixture
def valid_record() -> dict:
    return {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "timestamp": "2023-06-15 12:00:00",
        "open": 30000.0,
        "high": 30500.0,
        "low": 29800.0,
        "close": 30200.0,
        "volume": 1234.5,
    }


@pytest.fixture
def valid_records(valid_record) -> list:
    rec2 = {**valid_record, "timestamp": "2023-06-15 13:00:00", "close": 30300.0}
    return [valid_record, rec2]


# ---------------------------------------------------------------------------
# validate_ohlcv_records_simple
# ---------------------------------------------------------------------------

class TestValidateOhlcvRecordsSimple:
    def test_valid_records(self, valid_records):
        assert RecordValidator.validate_ohlcv_records_simple(valid_records) is True

    def test_none_returns_false(self):
        assert RecordValidator.validate_ohlcv_records_simple(None) is False

    def test_empty_list_returns_false(self):
        assert RecordValidator.validate_ohlcv_records_simple([]) is False

    def test_string_returns_false(self):
        assert RecordValidator.validate_ohlcv_records_simple("not a list") is False

    def test_dict_instead_of_list_returns_false(self):
        assert RecordValidator.validate_ohlcv_records_simple({"a": 1}) is False

    def test_non_dict_item_returns_false(self, valid_record):
        records = [valid_record, "not a dict"]
        assert RecordValidator.validate_ohlcv_records_simple(records) is False

    def test_missing_required_field_returns_false(self, valid_record):
        rec = {k: v for k, v in valid_record.items() if k != "symbol"}
        assert RecordValidator.validate_ohlcv_records_simple([rec]) is False

    def test_invalid_numeric_field_returns_false(self, valid_record):
        rec = {**valid_record, "open": "not_a_number"}
        assert RecordValidator.validate_ohlcv_records_simple([rec]) is False

    def test_none_numeric_field_returns_false(self, valid_record):
        rec = {**valid_record, "high": None}
        assert RecordValidator.validate_ohlcv_records_simple([rec]) is False

    def test_invalid_timestamp_returns_false(self, valid_record):
        rec = {**valid_record, "timestamp": "invalid-date-xyz"}
        assert RecordValidator.validate_ohlcv_records_simple([rec]) is False

    def test_empty_timestamp_returns_false(self, valid_record):
        rec = {**valid_record, "timestamp": ""}
        assert RecordValidator.validate_ohlcv_records_simple([rec]) is False

    def test_numeric_timestamp_accepted(self, valid_record):
        """ミリ秒タイムスタンプも受け付ける"""
        rec = {**valid_record, "timestamp": 1686830400000}
        assert RecordValidator.validate_ohlcv_records_simple([rec]) is True

    def test_multiple_records_all_valid(self, valid_records):
        assert RecordValidator.validate_ohlcv_records_simple(valid_records) is True

    def test_multiple_records_one_bad(self, valid_record, valid_records):
        bad = {**valid_record, "volume": "bad"}
        records = valid_records + [bad]
        assert RecordValidator.validate_ohlcv_records_simple(records) is False


# ---------------------------------------------------------------------------
# sanitize_ohlcv_data
# ---------------------------------------------------------------------------

class TestSanitizeOhlcvData:
    def test_basic_sanitization(self, valid_records):
        result = RecordValidator.sanitize_ohlcv_data(valid_records)

        assert len(result) == len(valid_records)
        for rec in result:
            assert rec["symbol"].isupper()
            assert rec["timeframe"].islower()
            assert isinstance(rec["timestamp"], (pd.Timestamp, datetime))
            assert isinstance(rec["open"], float)
            assert isinstance(rec["close"], float)

    def test_symbol_normalized(self, valid_record):
        rec = {**valid_record, "symbol": "  btc/usdt  "}
        result = RecordValidator.sanitize_ohlcv_data([rec])
        assert result[0]["symbol"] == "BTC/USDT"

    def test_timeframe_normalized(self, valid_record):
        rec = {**valid_record, "timeframe": "  1H  "}
        result = RecordValidator.sanitize_ohlcv_data([rec])
        assert result[0]["timeframe"] == "1h"

    def test_invalid_timestamp_raises(self, valid_record):
        rec = {**valid_record, "timestamp": "not-a-date"}
        with pytest.raises(ValueError, match="サニタイズに失敗しました"):
            RecordValidator.sanitize_ohlcv_data([rec])

    def test_invalid_numeric_raises(self, valid_record):
        rec = {**valid_record, "close": "abc"}
        with pytest.raises(ValueError, match="サニタイズに失敗しました"):
            RecordValidator.sanitize_ohlcv_data([rec])

    def test_empty_input_returns_empty(self):
        assert RecordValidator.sanitize_ohlcv_data([]) == []

    def test_numeric_fields_converted_to_float(self, valid_record):
        rec = {**valid_record, "open": "30000", "volume": "500"}
        result = RecordValidator.sanitize_ohlcv_data([rec])
        assert isinstance(result[0]["open"], float)
        assert result[0]["open"] == 30000.0
        assert isinstance(result[0]["volume"], float)


# ---------------------------------------------------------------------------
# DataValidator エイリアス
# ---------------------------------------------------------------------------

class TestDataValidatorAlias:
    def test_alias_points_to_record_validator(self):
        assert DataValidator is RecordValidator

    def test_alias_methods_work(self, valid_records):
        assert DataValidator.validate_ohlcv_records_simple(valid_records) is True
        sanitized = DataValidator.sanitize_ohlcv_data(valid_records)
        assert len(sanitized) == len(valid_records)
