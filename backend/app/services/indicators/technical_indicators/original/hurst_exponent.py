"""Hurst Exponent."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from ...data_validation import handle_pandas_ta_errors, validate_series_params


@njit(parallel=True, cache=True)
def _njit_hurst_loop(prices: np.ndarray, win: int, max_lag: int) -> np.ndarray:
    n = len(prices)
    res = np.full(n, np.nan)
    for i in prange(win - 1, n):
        chunk = prices[i - win + 1 : i + 1]
        log_returns = np.log(chunk[1:] / chunk[:-1])
        m = len(log_returns)
        if m < max_lag + 2:
            continue
        lags = np.arange(2, min(max_lag + 1, m // 2))
        if len(lags) < 3:
            continue
        log_rs = np.zeros(len(lags))
        for j in range(len(lags)):
            lag = lags[j]
            rs_vals = np.zeros(m - lag + 1)
            for k in range(len(rs_vals)):
                segment = log_returns[k : k + lag]
                mean_seg = np.mean(segment)
                std_seg = np.std(segment)
                if std_seg < 1e-12:
                    rs_vals[k] = 1.0
                    continue
                cum_dev = np.cumsum(segment - mean_seg)
                r_range = np.max(cum_dev) - np.min(cum_dev)
                rs_vals[k] = r_range / std_seg
            log_rs[j] = np.log(np.mean(rs_vals))
        log_lags = np.log(lags.astype(np.float64))
        x_mean = np.mean(log_lags)
        y_mean = np.mean(log_rs)
        ss_xy = np.sum((log_lags - x_mean) * (log_rs - y_mean))
        ss_xx = np.sum((log_lags - x_mean) ** 2)
        if ss_xx < 1e-12:
            res[i] = 0.5
        else:
            hurst = ss_xy / ss_xx
            res[i] = hurst
    return res


@handle_pandas_ta_errors
def hurst_exponent(
    close: pd.Series,
    length: int = 100,
    max_lag: int = 20,
) -> pd.Series:
    """Hurst Exponent.

    Measures the long-term memory of a time series.
    H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random walk.
    """
    if length < 10:
        raise ValueError("length must be >= 10")
    if max_lag < 2:
        raise ValueError("max_lag must be >= 2")

    validation = validate_series_params(close, length, min_data_length=length)
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"HURST_{length}_{max_lag}",
        )

    result = _njit_hurst_loop(close.to_numpy(dtype=float), length, max_lag)
    return pd.Series(result, index=close.index, name=f"HURST_{length}_{max_lag}")
