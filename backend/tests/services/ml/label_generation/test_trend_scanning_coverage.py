"""
Extra coverage tests for ``trend_scanning``.

Targets the ``use_log_price=False`` branch and other untested edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.ml.label_generation.trend_scanning import (
    TrendScanning,
    _compute_window_t_value,
    _label_from_t_value,
)


class TestComputeWindowTValue:
    """Cover the inner ``_compute_window_t_value`` helper via direct calls."""

    def test_zero_residual_yields_zero_t_value(self) -> None:
        # All residual components zero => slope = 0 => t = 0
        t = _compute_window_t_value(
            sum_y=10.0,
            sum_yy=30.0,
            sum_xy=20.0,
            n_val=5.0,
            sum_x=10.0,
            sum_xx=30.0,
            denominator=1.0,
        )
        # Slope=0 => abs(slope) < 1e-14 => t_val = 0
        assert t == 0.0

    def test_strictly_positive_slope_returns_large_positive_t(self) -> None:
        # Construct a perfect positive linear trend
        # y = 2x (x from 0 to 4)
        x = np.arange(5, dtype=float)
        y = 2.0 * x
        sum_x = x.sum()
        sum_xx = (x * x).sum()
        sum_y = y.sum()
        sum_yy = (y * y).sum()
        sum_xy = (x * y).sum()
        n_val = float(len(x))
        denominator = n_val * sum_xx - sum_x * sum_x
        t = _compute_window_t_value(
            sum_y=sum_y,
            sum_yy=sum_yy,
            sum_xy=sum_xy,
            n_val=n_val,
            sum_x=sum_x,
            sum_xx=sum_xx,
            denominator=denominator,
        )
        # Perfect fit => very large positive t (clipped to 100)
        assert t > 0.0
        # Should be clipped to <= 100
        assert t <= 100.0

    def test_strictly_negative_slope_returns_large_negative_t(self) -> None:
        x = np.arange(5, dtype=float)
        y = -2.0 * x
        sum_x = x.sum()
        sum_xx = (x * x).sum()
        sum_y = y.sum()
        sum_yy = (y * y).sum()
        sum_xy = (x * y).sum()
        n_val = float(len(x))
        denominator = n_val * sum_xx - sum_x * sum_x
        t = _compute_window_t_value(
            sum_y=sum_y,
            sum_yy=sum_yy,
            sum_xy=sum_xy,
            n_val=n_val,
            sum_x=sum_x,
            sum_xx=sum_xx,
            denominator=denominator,
        )
        assert t < 0.0
        assert t >= -100.0

    def test_zero_sum_y_squared(self) -> None:
        # sum_yy - sum_y^2/n_val = 0 (i.e. all y identical) => slope=0
        t = _compute_window_t_value(
            sum_y=0.0,
            sum_yy=0.0,
            sum_xy=0.0,
            n_val=5.0,
            sum_x=10.0,
            sum_xx=30.0,
            denominator=1.0,
        )
        assert t == 0.0


class TestLabelFromTValue:
    def test_return_t_value_mode(self) -> None:
        assert _label_from_t_value(5.0, min_t_value=2.0, return_t_value_as_label=True) == 5.0
        assert _label_from_t_value(-3.0, min_t_value=2.0, return_t_value_as_label=True) == -3.0
        assert _label_from_t_value(0.0, min_t_value=2.0, return_t_value_as_label=True) == 0.0

    def test_discrete_labels(self) -> None:
        assert _label_from_t_value(5.0, min_t_value=2.0, return_t_value_as_label=False) == 1.0
        assert _label_from_t_value(-5.0, min_t_value=2.0, return_t_value_as_label=False) == -1.0
        assert _label_from_t_value(0.0, min_t_value=2.0, return_t_value_as_label=False) == 0.0
        # Strict boundary (t_val == min_t_value stays at 0)
        assert _label_from_t_value(2.0, min_t_value=2.0, return_t_value_as_label=False) == 0.0
        assert _label_from_t_value(-2.0, min_t_value=2.0, return_t_value_as_label=False) == 0.0
        # Just above / below the boundary
        assert _label_from_t_value(2.001, min_t_value=2.0, return_t_value_as_label=False) == 1.0
        assert _label_from_t_value(-2.001, min_t_value=2.0, return_t_value_as_label=False) == -1.0


class TestTrendScanningLinearMode:
    """The ``use_log_price=False`` branch was previously uncovered."""

    def test_linear_mode_works(self) -> None:
        dates = pd.date_range("2023-01-01", periods=30, freq="h")
        s = pd.Series(np.arange(30, dtype=float) * 1.5 + 100, index=dates)
        ts = TrendScanning(min_window=5, max_window=10)
        labels = ts.get_labels(s, use_log_price=False)
        assert not labels.empty
        # Strong uptrend => bin == 1
        assert (labels["bin"] == 1).all()

    def test_linear_mode_descending(self) -> None:
        dates = pd.date_range("2023-01-01", periods=30, freq="h")
        s = pd.Series(200.0 - np.arange(30, dtype=float) * 1.5, index=dates)
        ts = TrendScanning(min_window=5, max_window=10)
        labels = ts.get_labels(s, use_log_price=False)
        assert not labels.empty
        assert (labels["bin"] == -1).all()

    def test_linear_mode_constant(self) -> None:
        dates = pd.date_range("2023-01-01", periods=20, freq="h")
        s = pd.Series(100.0, index=dates)
        ts = TrendScanning(min_window=5, max_window=10)
        labels = ts.get_labels(s, use_log_price=False)
        # Constant price => no trend => bin = 0
        assert not labels.empty
        assert (labels["bin"] == 0).all()


class TestTrendScanningEdgeCases:
    def test_empty_close_index(self) -> None:
        # Empty close with empty t_events
        ts = TrendScanning()
        s = pd.Series([], dtype=float)
        labels = ts.get_labels(s)
        assert labels.empty

    def test_t_events_subset(self) -> None:
        # Use only some timestamps
        dates = pd.date_range("2023-01-01", periods=50, freq="h")
        s = pd.Series(np.arange(50, dtype=float), index=dates)
        ts = TrendScanning(min_window=5, max_window=10)
        # Use only first 5 timestamps
        labels = ts.get_labels(s, t_events=dates[:5])
        assert len(labels) == 5

    def test_return_columns(self) -> None:
        dates = pd.date_range("2023-01-01", periods=20, freq="h")
        s = pd.Series(np.arange(20, dtype=float), index=dates)
        ts = TrendScanning(min_window=5, max_window=10)
        labels = ts.get_labels(s)
        # Required columns
        for col in ["t1", "t_value", "bin", "ret"]:
            assert col in labels.columns
