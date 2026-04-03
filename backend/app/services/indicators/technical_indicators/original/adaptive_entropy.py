"""Adaptive Entropy Oscillator."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit, prange

from ...data_validation import handle_pandas_ta_errors, validate_series_params
from ._window_helpers import _window_min_max


@njit(parallel=True, cache=True)
def _njit_entropy_loop(data: np.ndarray, window: int) -> np.ndarray:
    n = len(data)
    result = np.full(n, np.nan)
    if n < window:
        return result

    for i in prange(window - 1, n):
        win_data = data[i - window + 1 : i + 1]
        d_min, d_max = _window_min_max(win_data, 0, len(win_data))

        if d_min == d_max:
            result[i] = 0.0
            continue

        n_bins = min(10, window)
        hist = np.zeros(n_bins)
        bin_w = (d_max - d_min) / n_bins

        for val in win_data:
            b_idx = int((val - d_min) / bin_w)
            if b_idx >= n_bins:
                b_idx = n_bins - 1
            hist[b_idx] += 1

        entropy = 0.0
        inv_total = 1.0 / window
        for count in hist:
            p = count * inv_total
            if p > 1e-12:
                entropy -= p * np.log(p)
        result[i] = entropy
    return result


@handle_pandas_ta_errors
def adaptive_entropy(
    close: pd.Series,
    short_length: int = 14,
    long_length: int = 28,
    signal_length: int = 5,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Adaptive Entropy Oscillator."""
    if short_length < 5:
        raise ValueError("short_length must be >= 5")
    if long_length < 10:
        raise ValueError("long_length must be >= 10")
    if signal_length < 2:
        raise ValueError("signal_length must be >= 2")
    if short_length >= long_length:
        raise ValueError("short_length must be < long_length")

    validation = validate_series_params(close, long_length, min_data_length=long_length)
    if validation is not None:
        nan_osc = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"ADAPTIVE_ENTROPY_OSC_{short_length}_{long_length}",
        )
        nan_sig = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=(
                f"ADAPTIVE_ENTROPY_SIGNAL_{short_length}_{long_length}_{signal_length}"
            ),
        )
        nan_ratio = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"ADAPTIVE_ENTROPY_RATIO_{short_length}_{long_length}",
        )
        return nan_osc, nan_sig, nan_ratio

    prices = close.astype(float).to_numpy()
    short_entropy = _njit_entropy_loop(prices, short_length)
    long_entropy = _njit_entropy_loop(prices, long_length)

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_ratio = short_entropy / long_entropy

    normalized_osc = (entropy_ratio - 0.5) * 2.0

    signal = (
        pd.Series(normalized_osc, index=close.index)
        .rolling(window=signal_length)
        .mean()
    )

    oscillator = pd.Series(
        normalized_osc,
        index=close.index,
        name=f"ADAPTIVE_ENTROPY_OSC_{short_length}_{long_length}",
    )
    signal.name = (  # type: ignore[reportAttributeAccessIssue]
        f"ADAPTIVE_ENTROPY_SIGNAL_{short_length}_{long_length}_{signal_length}"
    )
    ratio = pd.Series(
        entropy_ratio,
        index=close.index,
        name=f"ADAPTIVE_ENTROPY_RATIO_{short_length}_{long_length}",
    )

    return oscillator, signal, ratio
