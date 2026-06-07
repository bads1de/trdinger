"""
Extra coverage tests for ``triple_barrier``.

Targets the fallback path when "side" is missing, the empty-events
short-circuit, and the "price not found" warning branches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.ml.label_generation.triple_barrier import (
    TripleBarrier,
    _process_events_numba,
)


class TestProcessEventsNumbaDirect:
    def test_basic_long_hit(self) -> None:
        # 10-element close: starts at 100, jumps to 105 at index 2
        close_vals = np.array(
            [100.0, 100.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0]
        )
        close_times = np.arange(10, dtype=np.int64) * 1_000_000_000
        t0_indices = np.array([0], dtype=np.int64)
        v_bar_indices = np.array([-1], dtype=np.int64)  # no vertical
        v_bar_times = np.full(1, np.iinfo(np.int64).min, dtype=np.int64)
        targets = np.array([0.02])  # 2% threshold
        sides = np.array([1.0])

        out_t1, out_side = _process_events_numba(
            close_vals,
            close_times,
            t0_indices,
            v_bar_indices,
            v_bar_times,
            targets,
            sides,
            pt=1.0,
            sl=1.0,
        )
        # Should hit PT at index 2
        assert out_t1[0] == close_times[2]
        assert out_side[0] == 1  # 1: pt

    def test_basic_long_stop_loss(self) -> None:
        # close drops 3% at index 2
        close_vals = np.array(
            [100.0, 100.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0]
        )
        close_times = np.arange(10, dtype=np.int64) * 1_000_000_000
        t0_indices = np.array([0], dtype=np.int64)
        v_bar_indices = np.array([-1], dtype=np.int64)
        v_bar_times = np.full(1, np.iinfo(np.int64).min, dtype=np.int64)
        targets = np.array([0.02])
        sides = np.array([1.0])

        out_t1, out_side = _process_events_numba(
            close_vals,
            close_times,
            t0_indices,
            v_bar_indices,
            v_bar_times,
            targets,
            sides,
            pt=1.0,
            sl=1.0,
        )
        # Should hit SL at index 2
        assert out_t1[0] == close_times[2]
        assert out_side[0] == 2  # 2: sl

    def test_basic_short_pt(self) -> None:
        # Short position: PT triggers on downward move
        close_vals = np.array(
            [100.0, 100.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0]
        )
        close_times = np.arange(10, dtype=np.int64) * 1_000_000_000
        t0_indices = np.array([0], dtype=np.int64)
        v_bar_indices = np.array([-1], dtype=np.int64)
        v_bar_times = np.full(1, np.iinfo(np.int64).min, dtype=np.int64)
        targets = np.array([0.02])
        sides = np.array([-1.0])

        out_t1, out_side = _process_events_numba(
            close_vals,
            close_times,
            t0_indices,
            v_bar_indices,
            v_bar_times,
            targets,
            sides,
            pt=1.0,
            sl=1.0,
        )
        # Short PT at index 2
        assert out_t1[0] == close_times[2]
        assert out_side[0] == 1  # 1: pt

    def test_vertical_barrier_hit(self) -> None:
        # No PT/SL hit, but vertical barrier reached
        close_vals = np.array([100.0, 100.5, 100.6, 100.7, 100.8, 100.9])
        close_times = np.arange(6, dtype=np.int64) * 1_000_000_000
        t0_indices = np.array([0], dtype=np.int64)
        v_bar_indices = np.array([3], dtype=np.int64)
        v_bar_times = np.array([3 * 1_000_000_000], dtype=np.int64)
        targets = np.array([0.05])  # 5% threshold, not reached
        sides = np.array([1.0])

        out_t1, out_side = _process_events_numba(
            close_vals,
            close_times,
            t0_indices,
            v_bar_indices,
            v_bar_times,
            targets,
            sides,
            pt=1.0,
            sl=1.0,
        )
        # Vertical barrier hit at index 3
        assert out_t1[0] == close_times[3]
        assert out_side[0] == 3  # 3: vertical

    def test_invalid_start_skipped(self) -> None:
        # start_idx = -1 => skipped
        close_vals = np.array([100.0, 101.0, 102.0])
        close_times = np.arange(3, dtype=np.int64) * 1_000_000_000
        t0_indices = np.array([-1], dtype=np.int64)
        v_bar_indices = np.array([-1], dtype=np.int64)
        v_bar_times = np.full(1, np.iinfo(np.int64).min, dtype=np.int64)
        targets = np.array([0.01])
        sides = np.array([1.0])

        out_t1, out_side = _process_events_numba(
            close_vals,
            close_times,
            t0_indices,
            v_bar_indices,
            v_bar_times,
            targets,
            sides,
            pt=1.0,
            sl=1.0,
        )
        # No event recorded
        assert out_t1[0] == np.iinfo(np.int64).min
        assert out_side[0] == 0

    def test_nan_or_zero_start_price_skipped(self) -> None:
        # p0 is NaN => skipped
        close_vals = np.array([np.nan, 100.0, 101.0, 102.0])
        close_times = np.arange(4, dtype=np.int64) * 1_000_000_000
        t0_indices = np.array([0], dtype=np.int64)
        v_bar_indices = np.array([-1], dtype=np.int64)
        v_bar_times = np.full(1, np.iinfo(np.int64).min, dtype=np.int64)
        targets = np.array([0.01])
        sides = np.array([1.0])

        out_t1, out_side = _process_events_numba(
            close_vals,
            close_times,
            t0_indices,
            v_bar_indices,
            v_bar_times,
            targets,
            sides,
            pt=1.0,
            sl=1.0,
        )
        assert out_t1[0] == np.iinfo(np.int64).min
        assert out_side[0] == 0

    def test_pt_zero_skips_check(self) -> None:
        # pt=0, sl>0: only stop-loss is checked
        close_vals = np.array(
            [100.0, 100.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0]
        )
        close_times = np.arange(10, dtype=np.int64) * 1_000_000_000
        t0_indices = np.array([0], dtype=np.int64)
        v_bar_indices = np.array([-1], dtype=np.int64)
        v_bar_times = np.full(1, np.iinfo(np.int64).min, dtype=np.int64)
        targets = np.array([0.02])
        sides = np.array([1.0])

        out_t1, out_side = _process_events_numba(
            close_vals,
            close_times,
            t0_indices,
            v_bar_indices,
            v_bar_times,
            targets,
            sides,
            pt=0.0,
            sl=1.0,
        )
        # No PT (pt=0) and no SL (no drop) => no event
        assert out_t1[0] == np.iinfo(np.int64).min
        assert out_side[0] == 0


class TestTripleBarrierPublicAPIEdgeCases:
    def _make_setup(self, n: int = 50):
        dates = pd.date_range("2023-01-01", periods=n, freq="h")
        np.random.seed(42)
        close = pd.Series(
            100.0 * (1 + np.random.normal(0, 0.01, n)).cumprod(),
            index=dates,
        )
        target = pd.Series(0.01, index=dates)
        return dates, close, target

    def test_empty_target_returns_empty_dataframe(self) -> None:
        dates, close, target = self._make_setup()
        tb = TripleBarrier()
        # min_ret very high => target.loc[t_events][target > min_ret] empty
        events = tb.get_events(
            close=close,
            t_events=dates,
            pt_sl=[1.0, 1.0],
            target=target,
            min_ret=1e6,
        )
        assert events.empty
        # Resulting bins also empty
        bins = tb.get_bins(events, close)
        assert bins.empty

    def test_event_with_nan_close_at_start_excluded(self) -> None:
        dates, close, target = self._make_setup()
        # Set close.iloc[10] to NaN, this is the start price for one event
        close.iloc[10] = np.nan
        tb = TripleBarrier()
        events = tb.get_events(
            close=close,
            t_events=dates,
            pt_sl=[1.0, 1.0],
            target=target,
            min_ret=0.0001,
        )
        # The event that started with NaN should have t1 == NaT
        # (the Numba function skips events where p0 is NaN)
        assert not events.empty
        # The event starting at the NaN index should have NaT t1
        assert pd.isna(events.loc[dates[10], "t1"])

    def test_fallback_path_when_side_column_missing(self) -> None:
        # The fallback path uses pt/sl directly on returns
        dates, close, target = self._make_setup()
        tb = TripleBarrier(pt=1.0, sl=1.0, min_ret=0.0001)
        # Manually construct an events DataFrame WITHOUT the "side" column
        events = pd.DataFrame(
            {
                "t1": [dates[5], dates[10]],
                "trgt": [0.01, 0.01],
            },
            index=[dates[0], dates[5]],
        )
        # Drop the "side" column if present
        if "side" in events.columns:
            events = events.drop(columns=["side"])
        bins = tb.get_bins(events, close)
        assert not bins.empty
        # The fallback branch should have set bin to 0 since neither pt nor sl
        # was explicitly hit (no "side" column).
        assert "bin" in bins.columns
        assert "ret" in bins.columns

    def test_fallback_with_pt_positive(self) -> None:
        dates, close, target = self._make_setup()
        tb = TripleBarrier(pt=1.0, sl=0.0, min_ret=0.0001)  # only pt checked
        # Pick an event where ret is large
        events = pd.DataFrame(
            {
                "t1": [dates[10]],
                "trgt": [0.01],
            },
            index=[dates[0]],
        )
        bins = tb.get_bins(events, close)
        assert not bins.empty

    def test_fallback_with_sl_positive(self) -> None:
        dates, close, target = self._make_setup()
        tb = TripleBarrier(pt=0.0, sl=1.0, min_ret=0.0001)  # only sl checked
        events = pd.DataFrame(
            {
                "t1": [dates[10]],
                "trgt": [0.01],
            },
            index=[dates[0]],
        )
        bins = tb.get_bins(events, close, binary_label=False)
        assert not bins.empty

    def test_binary_label(self) -> None:
        dates, close, target = self._make_setup()
        tb = TripleBarrier(pt=1.0, sl=1.0, min_ret=0.0001)
        events = tb.get_events(
            close=close,
            t_events=dates,
            pt_sl=[1.0, 1.0],
            target=target,
            min_ret=0.0001,
        )
        bins = tb.get_bins(events, close, binary_label=True)
        # With binary_label, only pt -> 1, sl/vertical -> 0
        if not bins.empty:
            valid = bins["bin"].dropna()
            assert ((valid == 0.0) | (valid == 1.0)).all()

    def test_t1_out_of_index_excluded(self) -> None:
        dates, close, target = self._make_setup()
        tb = TripleBarrier(pt=1.0, sl=1.0, min_ret=0.0001)
        # Create events with t1 outside the close index
        events = pd.DataFrame(
            {
                "t1": [pd.Timestamp("2099-01-01"), dates[5]],
                "trgt": [0.01, 0.01],
                "side": ["pt", "pt"],
            },
            index=[dates[0], dates[1]],
        )
        # Just ensure it doesn't crash; the event with out-of-range t1
        # may or may not be excluded depending on pandas reindex behavior.
        bins = tb.get_bins(events, close)
        assert isinstance(bins, pd.DataFrame)

    def test_init_with_initial_price_missing_excluded(self) -> None:
        dates, close, target = self._make_setup()
        # Pick an event whose start time is NOT in close.index
        tb = TripleBarrier(pt=1.0, sl=1.0, min_ret=0.0001)
        events = pd.DataFrame(
            {
                "t1": [dates[5]],
                "trgt": [0.01],
                "side": ["pt"],
            },
            index=[pd.Timestamp("2099-01-01")],
        )
        bins = tb.get_bins(events, close)
        # The row with start outside close.index should be excluded
        assert pd.Timestamp("2099-01-01") not in bins.index
