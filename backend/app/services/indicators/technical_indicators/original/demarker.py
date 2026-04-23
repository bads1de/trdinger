"""DeMarker Indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
)


@njit(cache=True)
def _njit_demarker_loop(
    high_values: np.ndarray,
    low_values: np.ndarray,
    length: int,
) -> np.ndarray:
    n = len(high_values)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length + 1:
        return result

    demax = np.zeros(n, dtype=np.float64)
    demin = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        high_delta = high_values[i] - high_values[i - 1]
        low_delta = low_values[i - 1] - low_values[i]

        if high_delta > 0.0:
            demax[i] = high_delta
        if low_delta > 0.0:
            demin[i] = low_delta

    rolling_max = 0.0
    rolling_min = 0.0

    for i in range(1, n):
        rolling_max += demax[i]
        rolling_min += demin[i]

        if i > length:
            rolling_max -= demax[i - length]
            rolling_min -= demin[i - length]

        if i >= length:
            denominator = rolling_max + rolling_min
            if denominator > 1e-12:
                result[i] = (rolling_max / denominator) * 100.0
            else:
                result[i] = 50.0

    return result


@handle_pandas_ta_errors
def demarker(
    high: pd.Series,
    low: pd.Series,
    length: int = 14,
) -> pd.Series:
    """DeMarker Indicator."""
    if length < 1:
        raise ValueError("length must be >= 1")

    validation = validate_multi_series_params(
        {"high": high, "low": low},
        length,
        min_data_length=length + 1,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(high), np.nan),
            index=high.index,
            name=f"DEMARKER_{length}",
        )

    result = _njit_demarker_loop(
        high.to_numpy(dtype=float),
        low.to_numpy(dtype=float),
        length,
    )
    return pd.Series(result, index=high.index, name=f"DEMARKER_{length}")
