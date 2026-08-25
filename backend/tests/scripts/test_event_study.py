"""イベントスタディ純粋関数のテスト."""

import pandas as pd
import pytest

from scripts.event_study import (
    align_events_to_hours,
    dedup_events,
    detect_fr_extremes,
    detect_lsr_extremes,
    detect_oi_flush,
    forward_returns,
    summarize_cell,
    trailing_percentile_flags,
)


def _series(
    values: list[float], start: str = "2024-01-01", freq: str = "h"
) -> pd.Series:
    idx = pd.DatetimeIndex(pd.date_range(start, periods=len(values), freq=freq))
    return pd.Series(values, index=idx, dtype=float)


class TestTrailingPercentileFlags:
    def test_dip_at_end_flagged_low(self):
        vals = [100.0] * 400 + [50.0] * 10
        low, high = trailing_percentile_flags(_series(vals), 3, 5.0)
        assert bool(low.tail(10).any())
        assert not bool(high.tail(10).any())

    def test_spike_flagged_high(self):
        vals = [100.0] * 400 + [200.0] * 10
        low, high = trailing_percentile_flags(_series(vals), 3, 5.0)
        assert bool(high.tail(10).any())

    def test_insufficient_history_no_flag(self):
        s = _series([1.0] * 20)
        low, high = trailing_percentile_flags(s, 30, 5.0)
        assert not low.any()
        assert not high.any()


class TestDetectFrExtremes:
    def test_negative_spike_detected(self):
        # FR実データと同じ8h間隔。90日分(270本)以上を用意
        base = [0.0001] * 400
        fr = _series(base + [-0.001] * 5 + base, freq="8h")
        h1, h2 = detect_fr_extremes(fr)
        assert len(h1) >= 1
        assert all(v < 0 for v in fr.loc[h1])
        assert len(h2) == 0

    def test_positive_spike_detected(self):
        base = [0.0001] * 400
        fr = _series(base + [0.002] * 5 + base, freq="8h")
        h1, h2 = detect_fr_extremes(fr)
        assert len(h2) >= 1
        assert len(h1) == 0


class TestDetectOiFlush:
    def test_flush_detected(self):
        oi = _series([1_000_000.0] * 100)
        oi.iloc[50:] = 800_000.0  # −20%
        events = detect_oi_flush(oi)
        assert len(events) >= 1

    def test_mild_decline_not_detected(self):
        oi = _series([1_000_000.0] * 60 + [900_000.0] * 40)  # −10%
        assert len(detect_oi_flush(oi)) == 0


class TestDetectLsrExtremes:
    def test_extremes_found_in_synthetic_data(self):
        # ベース0.55に約1%頻度で極端値を混ぜる
        vals = [0.55] * 1500
        for i in range(100, 1500, 100):
            vals[i] = 0.40 if i % 200 == 0 else 0.70
        lsr = _series(vals)
        low, high = detect_lsr_extremes(lsr)
        assert len(low) > 0 and len(high) > 0


class TestDedupEvents:
    def test_spacing_enforced(self):
        idx = pd.DatetimeIndex(
            pd.to_datetime(["2024-01-01 00:00", "2024-01-01 03:00", "2024-01-01 12:00"])
        )
        kept = dedup_events(idx, 8)
        assert list(kept) == [idx[0], idx[2]]

    def test_empty(self):
        assert len(dedup_events(pd.DatetimeIndex([]), 8)) == 0


class TestForwardReturns:
    def test_basic(self):
        closes = _series([100.0] * 30 + [110.0])
        fwd = forward_returns(closes, (24,))
        # idx6の24h後は idx30(=110)
        v = fwd[24].iloc[6]
        assert v == pytest.approx(0.10)

    def test_incomplete_window_is_nan(self):
        closes = _series([100.0] * 30)
        fwd = forward_returns(closes, (24,))
        assert pd.isna(fwd[24].iloc[10])


class TestAlignEvents:
    def test_maps_to_prior_hour_close(self):
        hourly_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=48, freq="h")
        )
        times = pd.DatetimeIndex([pd.Timestamp("2024-01-01 05:30")])
        aligned = align_events_to_hours(times, hourly_index)
        assert aligned[0] == pd.Timestamp("2024-01-01 05:00")

    def test_before_data_start_excluded(self):
        hourly_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=10, freq="h")
        )
        times = pd.DatetimeIndex([pd.Timestamp("2023-12-31 23:59")])
        assert len(align_events_to_hours(times, hourly_index)) == 0


class TestSummarizeCell:
    def test_stats(self):
        hours = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=4, freq="D"))
        fwd = pd.Series([0.01, -0.02, 0.03, 0.04], index=hours)
        cell = summarize_cell(hours, fwd, baseline_median=0.0)
        assert cell is not None
        assert cell["n"] == 4
        assert cell["win_rate"] == 75.0
        assert cell["median_pp"] == pytest.approx(2.0)

    def test_empty_returns_none(self):
        empty_idx = pd.DatetimeIndex([])
        assert summarize_cell(empty_idx, pd.Series(dtype=float), 0.0) is None
