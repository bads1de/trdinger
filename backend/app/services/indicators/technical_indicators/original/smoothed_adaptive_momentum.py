"""Smoothed Adaptive Momentum (SAM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_series_params


@njit(cache=True)
def _njit_sam_loop(
    prices: np.ndarray,
    length: int,
    smooth_length: int,
) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length + smooth_length:
        return result

    # Compute log returns
    log_ret = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if prices[i] > 1e-12 and prices[i - 1] > 1e-12:
            log_ret[i] = np.log(prices[i] / prices[i - 1])

    # Compute volatility (std of log returns over length)
    vol = np.full(n, np.nan, dtype=np.float64)
    for i in range(length, n):
        mean = 0.0
        for j in range(i - length + 1, i + 1):
            mean += log_ret[j]
        mean /= float(length)
        var = 0.0
        for j in range(i - length + 1, i + 1):
            diff = log_ret[j] - mean
            var += diff * diff
        vol[i] = np.sqrt(var / float(length))

    # Momentum = close - close[length] scaled by volatility
    mom = np.full(n, np.nan, dtype=np.float64)
    for i in range(length, n):
        raw_mom = prices[i] - prices[i - length]
        if vol[i] > 1e-12:
            mom[i] = raw_mom / (vol[i] * prices[i] + 1e-12) * 100.0
        else:
            mom[i] = 0.0

    # Smooth with SMA
    for i in range(length + smooth_length - 1, n):
        s = 0.0
        for j in range(i - smooth_length + 1, i + 1):
            s += mom[j]
        result[i] = s / float(smooth_length)

    return result


@handle_pandas_ta_errors
def smoothed_adaptive_momentum(
    close: pd.Series,
    length: int = 14,
    smooth_length: int = 5,
) -> pd.Series:
    """Smoothed Adaptive Momentum (SAM).

    ボラティリティで正規化したモメンタムを平滑化した指標。
    価格変動をボラティリティ基準でスケーリングし、
    SMAで平滑化することでノイズを低減する。
    """
    if length < 2:
        raise ValueError("length must be >= 2")
    if smooth_length < 1:
        raise ValueError("smooth_length must be >= 1")

    validation = validate_series_params(
        close,
        length,
        min_data_length=length + smooth_length,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"SAM_{length}_{smooth_length}",
        )

    result = _njit_sam_loop(
        close.to_numpy(dtype=float),
        length,
        smooth_length,
    )
    return pd.Series(
        result,
        index=close.index,
        name=f"SAM_{length}_{smooth_length}",
    )
