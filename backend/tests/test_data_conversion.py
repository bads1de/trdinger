"""データ変換モジュールのテスト

OHLCV、ファンディングレート、オープンインタレストの変換機能をテスト
エラーケースを追加してバグを洗い出す
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from app.utils.data_conversion import (
    OHLCVDataConverter,
    FundingRateDataConverter,
    OpenInterestDataConverter,
    ensure_list,
    DataConversionError
)


class TestOHLCVDataConverter:
    """OHLCVデータ変換テスト"""

    def test_ccxt_to_db_format_normal(self):
        """正常なCCXT形式のOHLCVデータを変換"""
        ohlcv_data = [
            [1638360000000, 50000.0, 51000.0, 49000.0, 50500.0, 100.5],  # ミリ秒タイムスタンプ
            [1638360060000, 50500.0, 51500.0, 50000.0, 51000.0, 95.0]
        ]

        result = OHLCVDataConverter.ccxt_to_db_format(ohlcv_data, "BTC/USDT", "1h")

        assert len(result) == 2
        assert result[0]["symbol"] == "BTC/USDT"
        assert result[0]["timeframe"] == "1h"
        assert result[0]["open"] == 50000.0
        assert result[0]["high"] == 51000.0
        assert result[0]["low"] == 49000.0
        assert result[0]["close"] == 50500.0
        assert result[0]["volume"] == 100.5
        assert isinstance(result[0]["timestamp"], datetime)

    def test_ccxt_to_db_format_short_list(self):
        """CCXTデータが短い場合のエラー処理"""
        ohlcv_data = [[1638360000000, 50000.0, 51000.0]]  # 要素不足

        with pytest.raises(ValueError):  # unpackエラー
            OHLCVDataConverter.ccxt_to_db_format(ohlcv_data, "BTC/USDT", "1h")

    def test_ccxt_to_db_format_invalid_timestamp(self):
        """無効なタイムスタンプの処理"""
        ohlcv_data = [["invalid", 50000.0, 51000.0, 49000.0, 50500.0, 100.5]]

        with pytest.raises(ValueError):  # fromtimestampでエラー
            OHLCVDataConverter.ccxt_to_db_format(ohlcv_data, "BTC/USDT", "1h")

    def test_ccxt_to_db_format_string_values(self):
        """文字列値のfloat変換"""
        ohlcv_data = [[1638360000000, "50000", "51000", "49000", "50500", "100.5"]]

        result = OHLCVDataConverter.ccxt_to_db_format(ohlcv_data, "BTC/USDT", "1h")

        assert result[0]["open"] == 50000.0  # 文字列がfloatに変換されるはず

    def test_db_to_api_format_normal(self):
        """正常なDBレコードをAPI形式に変換"""
        db_records = [
            Mock(
                timestamp=datetime.fromtimestamp(1638360000, tz=timezone.utc),
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=100.5
            ),
            Mock(
                timestamp=datetime.fromtimestamp(1638360060, tz=timezone.utc),
                open=50500.0,
                high=51500.0,
                low=50000.0,
                close=51000.0,
                volume=95.0
            )
        ]

        result = OHLCVDataConverter.db_to_api_format(db_records)

        assert len(result) == 2
        assert result[0] == [1638360000000, 50000.0, 51000.0, 49000.0, 50500.0, 100.5]
        assert result[1] == [1638360060000, 50500.0, 51500.0, 50000.0, 51000.0, 95.0]

    def test_db_to_api_format_naive_timestamp(self):
        """タイムゾーンなしタイムスタンプの処理"""
        # timestampがnaiveの場合、timezone.utcなしでtimestamp()を呼ぶとエラー
        import datetime
        db_records = [
            Mock(
                timestamp=datetime.datetime.fromtimestamp(1638360000),  # naive
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=100.5
            )
        ]

        with pytest.raises(TypeError):  # naive datetimeにtimestamp()はエラー
            OHLCVDataConverter.db_to_api_format(db_records)


class TestFundingRateDataConverter:
    """ファンディングレートデータ変換テスト"""

    def test_ccxt_to_db_format_normal(self):
        """正常なファンディングレートデータを変換"""
        funding_data = [
            {
                "datetime": "2023-10-01T12:00:00.000Z",
                "fundingRate": 0.0001,
                "nextFundingDatetime": "2023-10-01T13:00:00.000Z"
            },
            {
                "datetime": 1638360000000,  # ミリ秒
                "fundingRate": "0.0002",
                "nextFundingDatetime": 1638363600000
            }
        ]

        result = FundingRateDataConverter.ccxt_to_db_format(funding_data, "BTC/USDT")

        assert len(result) == 2
        assert result[0]["symbol"] == "BTC/USDT"
        assert result[0]["funding_rate"] == 0.0001
        assert isinstance(result[0]["data_timestamp"], datetime)
        assert isinstance(result[0]["next_funding_timestamp"], datetime)

        assert result[1]["funding_rate"] == 0.0002

    def test_ccxt_to_db_format_no_datetime(self):
        """datetimeフィールドがない場合"""
        funding_data = [
            {
                "fundingRate": 0.0001
            }
        ]

        result = FundingRateDataConverter.ccxt_to_db_format(funding_data, "BTC/USDT")

        assert len(result) == 1
        assert result[0]["data_timestamp"] is None

    def test_ccxt_to_db_format_invalid_datetime_string(self):
        """無効なdatetime文字列"""
        funding_data = [
            {
                "datetime": "invalid-date",
                "fundingRate": 0.0001
            }
        ]

        result = FundingRateDataConverter.ccxt_to_db_format(funding_data, "BTC/USDT")

        # fromisoformatでエラーになるはず
        assert len(result) == 1
        # 実際にはValueErrorになるが、テストでは確認

    def test_ccxt_to_db_format_string_funding_rate(self):
        """文字列のfundingRate"""
        funding_data = [
            {
                "datetime": "2023-10-01T12:00:00.000Z",
                "fundingRate": "0.0001"  # 文字列
            }
        ]

        result = FundingRateDataConverter.ccxt_to_db_format(funding_data, "BTC/USDT")

        assert result[0]["funding_rate"] == 0.0001  # float変換

    def test_ccxt_to_db_format_no_next_funding(self):
        """nextFundingDatetimeがない場合"""
        funding_data = [
            {
                "datetime": "2023-10-01T12:00:00.000Z",
                "fundingRate": 0.0001
            }
        ]

        result = FundingRateDataConverter.ccxt_to_db_format(funding_data, "BTC/USDT")

        assert "next_funding_timestamp" not in result[0]


class TestOpenInterestDataConverter:
    """オープンインタレストデータ変換テスト"""

    def test_ccxt_to_db_format_normal(self):
        """正常なオープンインタレストデータを変換"""
        oi_data = [
            {
                "timestamp": 1638360000000,
                "openInterestAmount": 1000000.0
            },
            {
                "timestamp": "2023-10-01T12:00:00.000Z",
                "openInterest": 2000000.0  # 別のフィールド名
            }
        ]

        result = OpenInterestDataConverter.ccxt_to_db_format(oi_data, "BTC/USDT")

        assert len(result) == 2
        assert result[0]["symbol"] == "BTC/USDT"
        assert result[0]["open_interest_value"] == 1000000.0
        assert isinstance(result[0]["data_timestamp"], datetime)

        assert result[1]["open_interest_value"] == 2000000.0

    def test_ccxt_to_db_format_no_valid_fields(self):
        """有効なフィールドがない場合"""
        oi_data = [
            {
                "timestamp": 1638360000000,
                "someField": "value"
            }
        ]

        result = OpenInterestDataConverter.ccxt_to_db_format(oi_data, "BTC/USDT")

        assert len(result) == 0  # 値が見つからないのでスキップ

    def test_ccxt_to_db_format_zero_value(self):
        """値が0の場合"""
        oi_data = [
            {
                "timestamp": 1638360000000,
                "openInterestAmount": 0.0
            },
            {
                "timestamp": 1638360060000,
                "openInterestAmount": None
            }
        ]

        result = OpenInterestDataConverter.ccxt_to_db_format(oi_data, "BTC/USDT")

        assert len(result) == 0  # 0やNoneはスキップ

    def test_ccxt_to_db_format_invalid_timestamp_string(self):
        """無効なタイムスタンプ文字列"""
        oi_data = [
            {
                "timestamp": "invalid-timestamp",
                "openInterestAmount": 1000000.0
            }
        ]

        result = OpenInterestDataConverter.ccxt_to_db_format(oi_data, "BTC/USDT")

        # fromisoformatでエラーになる可能性

    def test_ccxt_to_db_format_multiple_fields(self):
        """複数のフィールドがある場合"""
        oi_data = [
            {
                "openInterestAmount": 1000000.0,
                "openInterest": 500000.0,
                "openInterestValue": 750000.0
            }
        ]

        result = OpenInterestDataConverter.ccxt_to_db_format(oi_data, "BTC/USDT")

        assert len(result) == 1
        assert result[0]["open_interest_value"] == 1000000.0  # 最初の優先


class TestEnsureList:
    """ensure_list関数のテスト"""

    def test_ensure_list_normal(self):
        """正常な値をlistに変換"""
        assert ensure_list([1, 2, 3]) == [1, 2, 3]
        assert ensure_list("hello") == ["h", "e", "l", "l", "o"]  # 文字列は文字のリストに

    def test_ensure_list_numpy_array(self):
        """numpy arrayの変換"""
        try:
            import numpy as np
            arr = np.array([1, 2, 3])
            result = ensure_list(arr)
            assert result == [1, 2, 3]
        except ImportError:
            pytest.skip("numpy not available")

    def test_ensure_list_pandas_series(self):
        """pandas Seriesの変換"""
        try:
            import pandas as pd
            series = pd.Series([1, 2, 3])
            result = ensure_list(series)
            assert result == [1, 2, 3]
        except ImportError:
            pytest.skip("pandas not available")

    def test_ensure_list_invalid_data_attribute(self):
        """data属性がlist変換できない場合"""
        class FakeArray:
            def __init__(self):
                self.data = set([1, 2, 3])  # setは変換できない

        # TypeErrorでもAttributeErrorでも空リストに戻るはず
        fake = FakeArray()
        assert ensure_list(fake) == []  # dataが変換できないので空リスト

    def test_ensure_list_raise_on_error_false(self):
        """raise_on_error=Falseの場合"""
        class BadIterable:
            def __iter__(self):
                raise RuntimeError("iteration failed")

        bad_obj = BadIterable()
        result = ensure_list(bad_obj, raise_on_error=False)
        assert result == []  # エラー時は空リスト

    def test_ensure_list_uniterable(self):
        """イテレートできないオブジェクト"""
        # Noneなどはlist()でエラーにならない
        assert ensure_list(None) == []

    @pytest.mark.parametrize("input_val, expected", [
        (42, [42]),
        ([], []),
        ({"k": "v"}, ['k']),  # dict_keysのlist [ 'k' ]
        ((1, 2), [1, 2]),  # tuple
        (None, []),  # Noneは空リスト
    ])
    def test_ensure_list_parametrized(self, input_val, expected):
        """パラメータ化したテスト"""
        assert ensure_list(input_val) == expected


if __name__ == "__main__":
    pytest.main([__file__])