"""Shared OHLCV volatility estimators for ML feature engineering."""

from __future__ import annotations

from typing import Mapping, cast

import numpy as np
import pandas as pd
from numba import njit, prange


def _validate_series_bundle(series_map: Mapping[str, pd.Series]) -> pd.Index:
    """Validate that the provided series are aligned."""
    items = list(series_map.items())
    if not items:
        raise ValueError("At least one series is required")

    _, first_series = items[0]
    index = first_series.index
    length = len(first_series)

    for _, series in items[1:]:
        if len(series) != length:
            raise ValueError("All series must have the same length")
        if not series.index.equals(index):
            raise ValueError("All series must have the same index")

    return index


def _to_float_array(series: pd.Series) -> np.ndarray:
    """Coerce a pandas Series to a float64 numpy array."""
    coerced = pd.to_numeric(series, errors="coerce")
    return cast(pd.Series, coerced).to_numpy(dtype=np.float64, copy=False)


@njit(parallel=True, cache=True)
def _njit_yang_zhang_loop(open_arr, high_arr, low_arr, close_arr, length):
    """Numba accelerated Yang-Zhang volatility loop."""
    n = len(open_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length + 1:
        return result

    log_oc = np.zeros(n)
    log_co = np.zeros(n)
    rs_term = np.zeros(n)

    for i in prange(1, n):
        if (
            open_arr[i] > 0
            and high_arr[i] > 0
            and low_arr[i] > 0
            and close_arr[i - 1] > 0
            and close_arr[i] > 0
        ):
            log_oc[i] = np.log(open_arr[i] / close_arr[i - 1])
            log_co[i] = np.log(close_arr[i] / open_arr[i])
            rs_term[i] = (
                np.log(high_arr[i] / close_arr[i])
                * np.log(high_arr[i] / open_arr[i])
            ) + (
                np.log(low_arr[i] / close_arr[i])
                * np.log(low_arr[i] / open_arr[i])
            )

    k = 0.34 / (1.34 + (length + 1) / (length - 1))

    for i in prange(length, n):
        s_oc1, s_oc2 = 0.0, 0.0
        s_co1, s_co2 = 0.0, 0.0
        s_rs = 0.0

        for j in range(i - length + 1, i + 1):
            v_oc = log_oc[j]
            v_co = log_co[j]
            s_oc1 += v_oc
            s_oc2 += v_oc * v_oc
            s_co1 += v_co
            s_co2 += v_co * v_co
            s_rs += rs_term[j]

        v_oc_final = (s_oc2 - (s_oc1 * s_oc1) / length) / (length - 1)
        v_co_final = (s_co2 - (s_co1 * s_co1) / length) / (length - 1)
        m_rs = s_rs / length

        yz_variance = v_oc_final + k * v_co_final + (1.0 - k) * m_rs
        if yz_variance > 0:
            result[i] = np.sqrt(yz_variance)
        else:
            result[i] = 0.0

    return result


@njit(parallel=True, cache=True)
def _njit_parkinson_loop(high_arr, low_arr, length):
    """Numba accelerated Parkinson volatility loop."""
    n = len(high_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return result

    const = 1.0 / (4.0 * np.log(2.0))
    log_hl_sq = np.zeros(n)
    for i in prange(n):
        if high_arr[i] > 0 and low_arr[i] > 0:
            log_hl_sq[i] = np.log(high_arr[i] / low_arr[i]) ** 2

    for i in prange(length - 1, n):
        s = 0.0
        for j in range(i - length + 1, i + 1):
            s += log_hl_sq[j]

        result[i] = np.sqrt(const * (s / length))

    return result


@njit(parallel=True, cache=True)
def _njit_garman_klass_loop(open_arr, high_arr, low_arr, close_arr, length):
    """Numba accelerated Garman-Klass volatility loop."""
    n = len(open_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return result

    const = 2.0 * np.log(2.0) - 1.0
    inst_var = np.zeros(n)
    for i in prange(n):
        if (
            open_arr[i] > 0
            and high_arr[i] > 0
            and low_arr[i] > 0
            and close_arr[i] > 0
        ):
            v1 = 0.5 * (np.log(high_arr[i] / low_arr[i]) ** 2)
            v2 = const * (np.log(close_arr[i] / open_arr[i]) ** 2)
            val = v1 - v2
            inst_var[i] = val if val > 0 else 0.0

    for i in prange(length - 1, n):
        s = 0.0
        for j in range(i - length + 1, i + 1):
            s += inst_var[j]
        result[i] = np.sqrt(s / length)

    return result


def yang_zhang_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute the Yang-Zhang volatility estimator."""
    if window <= 1:
        raise ValueError("window must be greater than 1")

    index = _validate_series_bundle(
        {"open": open_, "high": high, "low": low, "close": close}
    )
    values = _njit_yang_zhang_loop(
        _to_float_array(open_),
        _to_float_array(high),
        _to_float_array(low),
        _to_float_array(close),
        window,
    )
    return pd.Series(values, index=index, name=f"Yang_Zhang_Vol_{window}")


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute the Parkinson volatility estimator."""
    if window <= 0:
        raise ValueError("window must be positive")

    index = _validate_series_bundle({"high": high, "low": low})
    values = _njit_parkinson_loop(
        _to_float_array(high),
        _to_float_array(low),
        window,
    )
    return pd.Series(values, index=index, name=f"Parkinson_Vol_{window}")


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute the Garman-Klass volatility estimator."""
    if window <= 0:
        raise ValueError("window must be positive")

    index = _validate_series_bundle(
        {"open": open_, "high": high, "low": low, "close": close}
    )
    values = _njit_garman_klass_loop(
        _to_float_array(open_),
        _to_float_array(high),
        _to_float_array(low),
        _to_float_array(close),
        window,
    )
    return pd.Series(values, index=index, name=f"Garman_Klass_Vol_{window}")


__all__ = [
    "garman_klass_volatility",
    "parkinson_volatility",
    "yang_zhang_volatility",
]
