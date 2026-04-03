"""Kairi Relative Index (KRI)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_series_params
from ._window_helpers import _window_mean


@njit(cache=True)
def _njit_kairi_loop(prices, length):
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return result
    for i in range(length - 1, n):
        sma = _window_mean(prices, i - length + 1, i + 1)
        if abs(sma) > 1e-12:
            result[i] = ((prices[i] - sma) / sma) * 100.0
    return result


@handle_pandas_ta_errors
def kairi_relative_index(close, length=14, signal_length=3):
    """Kairi Relative Index (KRI).

    Percentage deviation of price from its moving average.
    Positive = above MA, negative = below MA.

    Args:
        close: Close price series. length: SMA period. Default 14.
    Returns:
        Tuple of (kri, signal).
    """
    validation = validate_series_params(close, length, min_data_length=length)
    if validation is not None:
        nan1 = pd.Series(
            np.full(len(close), np.nan), index=close.index, name=f"KRI_{length}"
        )
        nan2 = pd.Series(
            np.full(len(close), np.nan), index=close.index, name=f"KRI_SIGNAL_{length}"
        )
        return nan1, nan2

    result = _njit_kairi_loop(close.values.astype(float), length)
    osc = pd.Series(result, index=close.index, name=f"KRI_{length}")
    sig = osc.rolling(window=signal_length, min_periods=1).mean()
    sig.name = f"KRI_SIGNAL_{length}"  # type: ignore[reportAttributeAccessIssue]
    return osc, sig
