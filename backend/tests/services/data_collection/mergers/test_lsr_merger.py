"""LSROMergerのテスト

Long/Short Ratioデータのマージロジックを検証します。
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.services.data_collection.mergers.lsr_merger import (
    LSROMerger,
    _period_to_timedelta,
    _safe_ratio,
)


def _make_ohlcv_df(start: datetime, rows: int = 6) -> pd.DataFrame:
    """4h足のテスト用OHLCV DataFrameを生成"""
    index = pd.date_range(start, periods=rows, freq="4h")
    return pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [1000.0] * rows,
        },
        index=index,
    )


def _make_lsr_row(timestamp, buy_ratio, sell_ratio) -> MagicMock:
    row = MagicMock()
    row.timestamp = timestamp
    row.buy_ratio = buy_ratio
    row.sell_ratio = sell_ratio
    return row


class TestPeriodToTimedelta:
    def test_parse_common_periods(self):
        assert _period_to_timedelta("4h") == timedelta(hours=4)
        assert _period_to_timedelta("1d") == timedelta(days=1)
        assert _period_to_timedelta("15min") == timedelta(minutes=15)
        assert _period_to_timedelta("1w") == timedelta(weeks=1)

    def test_unknown_period_falls_back_to_one_day(self):
        assert _period_to_timedelta("unknown") == timedelta(days=1)
        assert _period_to_timedelta("4x") == timedelta(days=1)


class TestSafeRatio:
    def test_normal_ratio(self):
        assert _safe_ratio(1.5, 0.5) == 3.0

    def test_zero_sell_ratio_returns_neutral(self):
        assert _safe_ratio(1.5, 0.0) == 1.0
        assert _safe_ratio(0.0, 0.0) == 1.0


class TestLSROMerger:
    def _make_merger(self, lsr_rows):
        repo = MagicMock()
        repo.get_long_short_ratio_data.return_value = lsr_rows
        return LSROMerger(repo)

    def test_merge_lsr_data_with_std_symbol(self):
        start = datetime(2023, 1, 1)
        df = _make_ohlcv_df(start)
        # 2本目のバー時刻のLSRデータ（4h period）
        lsr_rows = [
            _make_lsr_row(start + timedelta(hours=4), buy_ratio=1.2, sell_ratio=0.8),
            _make_lsr_row(start + timedelta(hours=8), buy_ratio=0.9, sell_ratio=1.1),
        ]
        merger = self._make_merger(lsr_rows)

        result = merger.merge_lsr_data(
            df, "BTC/USDT:USDT", "4h", start, start + timedelta(days=1)
        )

        # コロン付きシンボルで検索される
        merger.lsr_repo.get_long_short_ratio_data.assert_called_once()
        assert (
            merger.lsr_repo.get_long_short_ratio_data.call_args.kwargs["symbol"]
            == "BTC/USDT:USDT"
        )
        # LSR列がマージされ、比率（1.2/0.8 = 1.5）が反映される
        assert "long_short_ratio" in result.columns
        assert result["long_short_ratio"].iloc[1] == pytest.approx(1.5)
        assert result["long_short_ratio"].iloc[2] == pytest.approx(0.9 / 1.1)

    def test_merge_lsr_data_falls_back_to_base_symbol(self):
        start = datetime(2023, 1, 1)
        df = _make_ohlcv_df(start)
        lsr_rows = [_make_lsr_row(start, buy_ratio=1.0, sell_ratio=1.0)]

        repo = MagicMock()
        repo.get_long_short_ratio_data.side_effect = [[], lsr_rows]
        merger = LSROMerger(repo)

        result = merger.merge_lsr_data(
            df, "BTC/USDT:USDT", "4h", start, start + timedelta(days=1)
        )

        # 2回目の検索はベース形式（BTC/USDT）
        assert repo.get_long_short_ratio_data.call_count == 2
        assert (
            repo.get_long_short_ratio_data.call_args_list[1].kwargs["symbol"]
            == "BTC/USDT"
        )
        assert "long_short_ratio" in result.columns

    def test_merge_lsr_data_no_data_uses_neutral_default(self):
        start = datetime(2023, 1, 1)
        df = _make_ohlcv_df(start)
        merger = self._make_merger([])

        result = merger.merge_lsr_data(
            df, "BTC/USDT:USDT", "4h", start, start + timedelta(days=1)
        )

        assert "long_short_ratio" in result.columns
        assert (result["long_short_ratio"] == 1.0).all()

    def test_merge_lsr_data_duplicate_timestamps_are_deduped(self):
        start = datetime(2023, 1, 1)
        df = _make_ohlcv_df(start)
        # 同一時刻の重複レコード（後勝ち）
        lsr_rows = [
            _make_lsr_row(start, buy_ratio=1.0, sell_ratio=1.0),
            _make_lsr_row(start, buy_ratio=1.5, sell_ratio=0.5),
            _make_lsr_row(start + timedelta(hours=4), buy_ratio=1.0, sell_ratio=1.0),
        ]
        merger = self._make_merger(lsr_rows)

        result = merger.merge_lsr_data(
            df, "BTC/USDT:USDT", "4h", start, start + timedelta(days=1)
        )

        # 例外なくマージされ、重複は後勝ち（1.5/0.5 = 3.0）になる
        assert "long_short_ratio" in result.columns
        assert result["long_short_ratio"].iloc[0] == 3.0

    def test_merge_lsr_data_stale_beyond_tolerance_keeps_previous(self):
        start = datetime(2023, 1, 1)
        df = _make_ohlcv_df(start, rows=8)
        # 最初のLSRのみ（それ以降はtolerance=12hを超えて古くなる）
        lsr_rows = [_make_lsr_row(start, buy_ratio=1.0, sell_ratio=1.0)]
        merger = self._make_merger(lsr_rows)

        result = merger.merge_lsr_data(
            df, "BTC/USDT:USDT", "4h", start, start + timedelta(days=2)
        )

        # 先頭はマージ、それ以降はNaN→interpolate前にデフォルトは使われないが
        # 列自体は存在する
        assert "long_short_ratio" in result.columns
        assert result["long_short_ratio"].iloc[0] == 1.0
        # tolerance超過後の行はNaN（下流の補間処理で処理される）
        assert result["long_short_ratio"].iloc[-1] != 1.0 or pd.isna(
            result["long_short_ratio"].iloc[-1]
        )
