import pandas as pd
import pytest

from scripts.h3_execution_decay import (
    ROUND_TRIP_COST_PP,
    count_episodes,
    delayed_forward_returns,
)
from scripts.h3_threshold_sensitivity import detect_oi_flush_at


class TestDelayedForwardReturns:
    def test_zero_delay_matches_simple_ratio(self) -> None:
        idx = pd.date_range("2024-01-01", periods=48, freq="1h")
        closes = pd.Series([100.0 + i for i in range(48)], index=idx)
        fwd = delayed_forward_returns(closes, 0, 24)
        assert fwd.iloc[0] == pytest.approx((124 - 100) / 100)
        assert pd.isna(fwd.iloc[30])

    def test_delay_shifts_entry(self) -> None:
        idx = pd.date_range("2024-01-01", periods=48, freq="1h")
        closes = pd.Series([100.0 + i for i in range(48)], index=idx)
        fwd = delayed_forward_returns(closes, 4, 24)
        # t=0: entry at close[4]=104, exit close[28]=128
        assert fwd.iloc[0] == pytest.approx((128 - 104) / 104)

    def test_invalid_args_raise(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="1h")
        closes = pd.Series(100.0, index=idx)
        with pytest.raises(ValueError):
            delayed_forward_returns(closes, -1, 24)
        with pytest.raises(ValueError):
            delayed_forward_returns(closes, 1, 0)


class TestCountEpisodes:
    def test_empty(self) -> None:
        assert count_episodes(pd.DatetimeIndex([]), 72) == 0

    def test_gap_splitting(self) -> None:
        ts = pd.DatetimeIndex(
            [
                "2024-01-01 00:00",
                "2024-01-02 00:00",
                "2024-01-03 00:00",
                "2024-01-10 00:00",
            ]
        )
        assert count_episodes(ts, 72) == 2


def test_cost_constant_frozen() -> None:
    assert ROUND_TRIP_COST_PP == 0.08


class TestDetectOiFlushAt:
    def _oi(self) -> pd.Series:
        idx = pd.date_range("2024-01-01", periods=30, freq="1h")
        vals = [100.0] * 24 + [90.0] * 3 + [110.0] * 3
        return pd.Series(vals, index=idx)

    def test_threshold_selects_flush_hours(self) -> None:
        oi = self._oi()
        # t=24..26 have 24h change -10%
        got = detect_oi_flush_at(oi, -0.05)
        assert list(got) == list(oi.index[24:27])

    def test_stricter_threshold_excludes_mild_drop(self) -> None:
        oi = self._oi()
        got = detect_oi_flush_at(oi, -0.15)
        assert len(got) == 0

    def test_empty_series(self) -> None:
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        assert len(detect_oi_flush_at(empty, -0.15)) == 0
