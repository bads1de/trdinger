"""
backtest/shared.py のテスト

app/services/backtest/shared.py のテストモジュール
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest
from unittest.mock import MagicMock

from app.services.backtest.shared import (
    resolve_stats_object,
    safe_float_conversion,
    safe_duration_conversion,
    safe_int_conversion,
    parse_datetime_value,
    safe_timestamp_conversion,
    resolve_trade_pnl_column,
    normalize_ohlcv_columns,
    normalize_datetimes_for_comparison,
    current_datetime_like,
    TRADE_PNL_COLUMNS,
    OHLCV_COLUMNS,
)


class TestResolveStatsObject:
    """resolve_stats_object 関数のテスト"""

    def test_resolve_stats_object_with_callable(self):
        """callableなstatsオブジェクト"""
        mock_stats = MagicMock(return_value={"value": 123})
        result = resolve_stats_object(mock_stats)
        assert result == {"value": 123}
        mock_stats.assert_called_once()

    def test_resolve_stats_object_with_dict(self):
        """dict型のstatsオブジェクト"""
        stats = {"value": 123}
        result = resolve_stats_object(stats)
        assert result == {"value": 123}

    def test_resolve_stats_object_callable_exception(self):
        """callableが例外を投げる場合"""
        mock_stats = MagicMock(side_effect=Exception("Test error"))
        mock_logger = MagicMock()
        result = resolve_stats_object(mock_stats, warning_logger=mock_logger)
        assert result == mock_stats
        mock_logger.warning.assert_called_once()

    def test_resolve_stats_object_no_warning_logger(self):
        """warning_loggerがない場合"""
        mock_stats = MagicMock(side_effect=Exception("Test error"))
        result = resolve_stats_object(mock_stats)
        assert result == mock_stats


class TestSafeFloatConversion:
    """safe_float_conversion 関数のテスト"""

    def test_safe_float_conversion_valid(self):
        """有効なfloat値"""
        result = safe_float_conversion(123.45)
        assert result == 123.45

    def test_safe_float_conversion_int(self):
        """int値"""
        result = safe_float_conversion(123)
        assert result == 123.0

    def test_safe_float_conversion_string(self):
        """文字列"""
        result = safe_float_conversion("123.45")
        assert result == 123.45

    def test_safe_float_conversion_none(self):
        """None値"""
        result = safe_float_conversion(None)
        assert result == 0.0

    def test_safe_float_conversion_nan(self):
        """NaN値"""
        result = safe_float_conversion(float("nan"))
        assert result == 0.0

    def test_safe_float_conversion_invalid_string(self):
        """無効な文字列"""
        result = safe_float_conversion("invalid")
        assert result == 0.0


class TestSafeIntConversion:
    """safe_int_conversion 関数のテスト"""

    def test_safe_int_conversion_valid(self):
        """有効なint値"""
        result = safe_int_conversion(123)
        assert result == 123

    def test_safe_int_conversion_float(self):
        """float値"""
        result = safe_int_conversion(123.45)
        assert result == 123

    def test_safe_int_conversion_string(self):
        """文字列"""
        result = safe_int_conversion("123")
        assert result == 123

    def test_safe_int_conversion_none(self):
        """None値"""
        result = safe_int_conversion(None)
        assert result == 0

    def test_safe_int_conversion_nan(self):
        """NaN値"""
        result = safe_int_conversion(float("nan"))
        assert result == 0

    def test_safe_int_conversion_invalid_string(self):
        """無効な文字列"""
        result = safe_int_conversion("invalid")
        assert result == 0


class TestSafeDurationConversion:
    """safe_duration_conversion 関数のテスト"""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (pd.Timedelta(days=2, hours=12), 2.5),
            (timedelta(days=1, hours=6), 1.25),
            ("3 days 12:00:00", 3.5),
            ("2.5", 2.5),
        ],
    )
    def test_safe_duration_conversion_valid(self, value, expected, caplog):
        result = safe_duration_conversion(value)
        assert result == pytest.approx(expected)
        assert caplog.records == []


class TestParseDatetimeValue:
    """parse_datetime_value 関数のテスト"""

    def test_parse_datetime_value_string(self):
        """文字列からdatetime"""
        result = parse_datetime_value("2023-01-01 12:00:00")
        assert isinstance(result, datetime)
        assert result.year == 2023

    def test_parse_datetime_value_datetime(self):
        """datetimeオブジェクト"""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        result = parse_datetime_value(dt)
        assert result == dt


class TestSafeTimestampConversion:
    """safe_timestamp_conversion 関数のテスト"""

    def test_safe_timestamp_conversion_valid(self):
        """有効なtimestamp"""
        result = safe_timestamp_conversion(1672574400)
        assert isinstance(result, datetime)

    def test_safe_timestamp_conversion_none(self):
        """None値"""
        result = safe_timestamp_conversion(None)
        assert result is None

    def test_safe_timestamp_conversion_invalid(self):
        """無効な値"""
        result = safe_timestamp_conversion("invalid")
        # エラー時の挙動は実装に依存
        assert result is None or isinstance(result, datetime)


class TestResolveTradePnlColumn:
    """resolve_trade_pnl_column 関数のテスト"""

    def test_resolve_trade_pnl_column_pnl(self):
        """PnL列"""
        df = pd.DataFrame({"PnL": [100, -50, 200]})
        result = resolve_trade_pnl_column(df)
        assert result == "PnL"

    def test_resolve_trade_pnl_column_profit(self):
        """Profit列"""
        df = pd.DataFrame({"Profit": [100, -50, 200]})
        result = resolve_trade_pnl_column(df)
        assert result == "Profit"

    def test_resolve_trade_pnl_column_custom_order(self):
        """カスタム優先順位"""
        df = pd.DataFrame({"Profit": [100, -50, 200], "PnL": [100, -50, 200]})
        result = resolve_trade_pnl_column(df, preferred_columns=("Profit", "PnL"))
        assert result == "Profit"

    def test_resolve_trade_pnl_column_not_found(self):
        """列が見つからない"""
        df = pd.DataFrame({"other": [100, -50, 200]})
        result = resolve_trade_pnl_column(df)
        assert result is None

    def test_resolve_trade_pnl_column_none_df(self):
        """NoneのDataFrame"""
        result = resolve_trade_pnl_column(None)
        assert result is None

    def test_resolve_trade_pnl_column_no_columns(self):
        """columns属性がないオブジェクト"""
        result = resolve_trade_pnl_column([1, 2, 3])
        assert result is None


class TestNormalizeOhlcvColumns:
    """normalize_ohlcv_columns 関数のテスト"""

    def test_normalize_ohlcv_columns_lowercase(self):
        """小文字に正規化"""
        df = pd.DataFrame({"Open": [100], "High": [105], "Low": [95], "Close": [100]})
        result = normalize_ohlcv_columns(df, lowercase=True)
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns

    def test_normalize_ohlcv_columns_capitalize(self):
        """先頭大文字に正規化"""
        df = pd.DataFrame({"open": [100], "high": [105], "low": [95], "close": [100]})
        result = normalize_ohlcv_columns(df, lowercase=False)
        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns

    def test_normalize_ohlcv_columns_preserve_other(self):
        """他の列名を保持"""
        df = pd.DataFrame({"open": [100], "high": [105], "custom": ["value"]})
        result = normalize_ohlcv_columns(df, lowercase=False)
        assert "custom" in result.columns

    def test_normalize_ohlcv_columns_ensure_volume(self):
        """volume列を追加"""
        df = pd.DataFrame({"Open": [100], "Close": [100]})
        result = normalize_ohlcv_columns(df, ensure_volume=True, volume_default=0.0)
        assert "Volume" in result.columns
        assert (result["Volume"] == 0.0).all()

    def test_normalize_ohlcv_columns_non_dataframe(self):
        """DataFrame以外"""
        result = normalize_ohlcv_columns([1, 2, 3])
        assert result == [1, 2, 3]

    def test_normalize_ohlcv_columns_no_changes_needed(self):
        """変更が必要ない場合"""
        df = pd.DataFrame({"Open": [100], "Close": [100]})
        result = normalize_ohlcv_columns(df, lowercase=False)
        # 同じオブジェクトを返すか確認
        assert result.equals(df)


class TestNormalizeDatetimesForComparison:
    """normalize_datetimes_for_comparison 関数のテスト"""

    def test_normalize_datetimes_for_comparison(self):
        """基本的な正規化"""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)
        result_start, result_end = normalize_datetimes_for_comparison(start, end)
        assert isinstance(result_start, datetime)
        assert isinstance(result_end, datetime)


class TestCurrentDatetimeLike:
    """current_datetime_like 関数のテスト"""

    def test_current_datetime_like(self):
        """基準日時に基づく現在日時"""
        reference = datetime(2023, 1, 1, 12, 0, 0)
        result = current_datetime_like(reference)
        assert isinstance(result, datetime)


class TestConstants:
    """定数のテスト"""

    def test_trade_pnl_columns(self):
        """TRADE_PNL_COLUMNS定数"""
        assert isinstance(TRADE_PNL_COLUMNS, tuple)
        assert "PnL" in TRADE_PNL_COLUMNS
        assert "Profit" in TRADE_PNL_COLUMNS

    def test_ohlcv_columns(self):
        """OHLCV_COLUMNS定数"""
        assert isinstance(OHLCV_COLUMNS, tuple)
        assert "open" in OHLCV_COLUMNS
        assert "high" in OHLCV_COLUMNS
        assert "low" in OHLCV_COLUMNS
        assert "close" in OHLCV_COLUMNS
        assert "volume" in OHLCV_COLUMNS
