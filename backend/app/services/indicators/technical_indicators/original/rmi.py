"""Relative Momentum Index (RMI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_series_params


@njit(cache=True)
def _njit_rmi_loop(
    prices: np.ndarray,
    length: int,
    momentum: int,
) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length + momentum:
        return result

    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)

    for i in range(momentum, n):
        delta = prices[i] - prices[i - momentum]
        if delta > 0.0:
            gains[i] = delta
        elif delta < 0.0:
            losses[i] = -delta

    seed_index = momentum + length - 1
    avg_gain = 0.0
    avg_loss = 0.0

    for i in range(momentum, seed_index + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]

    avg_gain /= float(length)
    avg_loss /= float(length)

    if avg_loss <= 1e-12:
        result[seed_index] = 100.0 if avg_gain > 1e-12 else 50.0
    else:
        rs = avg_gain / avg_loss
        result[seed_index] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(seed_index + 1, n):
        avg_gain = ((avg_gain * (length - 1)) + gains[i]) / float(length)
        avg_loss = ((avg_loss * (length - 1)) + losses[i]) / float(length)

        if avg_loss <= 1e-12:
            result[i] = 100.0 if avg_gain > 1e-12 else 50.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result


@handle_pandas_ta_errors
def rmi(
    close: pd.Series,
    length: int = 14,
    momentum: int = 5,
) -> pd.Series:
    """Relative Momentum Index (RMI)."""
    if length < 2:
        raise ValueError("length must be >= 2")
    if momentum < 1:
        raise ValueError("momentum must be >= 1")

    validation = validate_series_params(
        close,
        length,
        min_data_length=length + momentum,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"RMI_{length}_{momentum}",
        )

    result = _njit_rmi_loop(
        close.to_numpy(dtype=float),
        length,
        momentum,
    )
    return pd.Series(
        result, index=close.index, name=f"RMI_{length}_{momentum}"
    )
