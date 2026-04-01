"""Fractal Adaptive Moving Average (FRAMA)."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numba import njit, prange

from ._window_helpers import _window_min_max
from ...data_validation import (
    handle_pandas_ta_errors,
    validate_series_params,
)

ALPHA_MIN: Final[float] = 0.01
ALPHA_MAX: Final[float] = 1.0


@njit(parallel=True, cache=True)
def _njit_frama_loop(
    prices: np.ndarray,
    length: int,
    half: int,
    log2: float,
    w: float,
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    n = len(prices)
    filt = np.full(n, np.nan)
    for i in prange(length - 1, n):
        n1_high, n1_low = _window_min_max(prices, i - length + 1, i - half + 1)
        n1 = (n1_high - n1_low) / half

        n2_high, n2_low = _window_min_max(prices, i - half + 1, i + 1)
        n2 = (n2_high - n2_low) / half

        n3_high, n3_low = _window_min_max(prices, i - length + 1, i + 1)
        n3 = (n3_high - n3_low) / length

        if n1 > 1e-9 and n2 > 1e-9 and n3 > 1e-9:
            dimen = (np.log(n1 + n2) - np.log(n3)) / log2
        else:
            dimen = 1.0

        alpha = np.exp(w * (dimen - 1.0))
        if alpha < alpha_min:
            alpha = alpha_min
        if alpha > alpha_max:
            alpha = alpha_max

        if i == length - 1:
            filt[i] = prices[i]
        else:
            prev_filt = filt[i - 1]
            if np.isfinite(prev_filt):
                filt[i] = alpha * prices[i] + (1.0 - alpha) * prev_filt
            else:
                filt[i] = prices[i]
    return filt


@handle_pandas_ta_errors
def frama(close: pd.Series, length: int = 16, slow: int = 200) -> pd.Series:
    """Fractal Adaptive Moving Average (FRAMA)."""
    if length < 4:
        length = 4
    if length % 2 != 0:
        length += 1
    if slow < 1:
        slow = 1

    validation = validate_series_params(close, length, min_data_length=length)
    if validation is not None:
        return pd.Series(np.full(len(close), np.nan), index=close.index, name="FRAMA")

    prices = close.astype(float).to_numpy()
    half = length // 2
    log2 = np.log(2.0)
    slow_float = float(slow)
    w = 2.303 * np.log(2.0 / (slow_float + 1.0))

    result = _njit_frama_loop(
        prices,
        length,
        half,
        log2,
        w,
        ALPHA_MIN,
        ALPHA_MAX,
    )

    return pd.Series(result, index=close.index, name="FRAMA")