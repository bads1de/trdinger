"""Chande Qstick."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_multi_series_params


@njit(cache=True)
def _njit_qstick_loop(
    close_values: np.ndarray,
    open_values: np.ndarray,
    length: int,
) -> np.ndarray:
    n = len(close_values)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return result

    # diff = close - open
    diff = np.zeros(n, dtype=np.float64)
    for i in range(n):
        diff[i] = close_values[i] - open_values[i]

    # SMA of diff
    window_sum = 0.0
    for i in range(length):
        window_sum += diff[i]
    result[length - 1] = window_sum / float(length)

    for i in range(length, n):
        window_sum += diff[i]
        window_sum -= diff[i - length]
        result[i] = window_sum / float(length)

    return result


@handle_pandas_ta_errors
def qstick(
    close: pd.Series,
    open_: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Chande Qstick.

    Close - Openの移動平均で、ローソク足の本体の傾向を測定する。
    正の値は買い圧力、負の値は売り圧力を示す。
    """
    if length < 1:
        raise ValueError("length must be >= 1")

    validation = validate_multi_series_params(
        {"close": close, "open": open_},
        length,
        min_data_length=length,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"QSTICK_{length}",
        )

    result = _njit_qstick_loop(
        close.to_numpy(dtype=float),
        open_.to_numpy(dtype=float),
        length,
    )
    return pd.Series(result, index=close.index, name=f"QSTICK_{length}")
