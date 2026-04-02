"""Polarized Fractal Efficiency (PFE)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_series_params


@njit(cache=True)
def _njit_pfe_loop(
    prices: np.ndarray,
    length: int,
    smoothing_length: int,
) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return result

    segment_lengths = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        segment_lengths[i] = np.sqrt(1.0 + diff * diff)

    cumulative = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        cumulative[i] = cumulative[i - 1] + segment_lengths[i]

    raw = np.full(n, np.nan, dtype=np.float64)

    for i in range(length - 1, n):
        start = i - length + 2
        denom = cumulative[i]
        if start > 1:
            denom -= cumulative[start - 1]

        if denom <= 1e-12:
            continue

        direction = prices[i] - prices[i - length + 1]
        numerator = np.sqrt((direction * direction) + float(length * length))
        value = 100.0 * numerator / denom

        if direction < 0.0:
            value = -value
        elif direction == 0.0:
            value = 0.0

        raw[i] = value

    first_valid = -1
    for i in range(n):
        if np.isfinite(raw[i]):
            first_valid = i
            break

    if first_valid == -1:
        return result

    alpha = 2.0 / (smoothing_length + 1.0)
    ema = raw[first_valid]
    result[first_valid] = ema

    for i in range(first_valid + 1, n):
        if np.isfinite(raw[i]):
            ema = alpha * raw[i] + (1.0 - alpha) * ema
        result[i] = ema

    return result


@handle_pandas_ta_errors
def pfe(
    close: pd.Series,
    length: int = 10,
    smoothing_length: int = 5,
) -> pd.Series:
    """Polarized Fractal Efficiency (PFE)."""
    if length < 2:
        raise ValueError("length must be >= 2")
    if smoothing_length < 1:
        raise ValueError("smoothing_length must be >= 1")

    validation = validate_series_params(
        close,
        length,
        min_data_length=length,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"PFE_{length}_{smoothing_length}",
        )

    result = _njit_pfe_loop(
        close.astype(float).to_numpy(),
        length,
        smoothing_length,
    )
    return pd.Series(
        result,
        index=close.index,
        name=f"PFE_{length}_{smoothing_length}",
    )
