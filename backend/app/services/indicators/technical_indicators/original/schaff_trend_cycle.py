"""Schaff Trend Cycle (STC)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_series_params


@njit(cache=True)
def _njit_ema_loop(prices: np.ndarray, length: int) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return result

    alpha = 2.0 / (length + 1.0)
    seed = 0.0
    for i in range(length):
        seed += prices[i]
    result[length - 1] = seed / float(length)

    for i in range(length, n):
        result[i] = alpha * prices[i] + (1.0 - alpha) * result[i - 1]

    return result


@njit(cache=True)
def _njit_stc_loop(
    prices: np.ndarray,
    fast_length: int,
    slow_length: int,
    cycle_length: int,
) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    min_len = slow_length + cycle_length
    if n < min_len:
        return result

    # MACD line = fast EMA - slow EMA
    fast_ema = _njit_ema_loop(prices, fast_length)
    slow_ema = _njit_ema_loop(prices, slow_length)

    macd = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(fast_ema[i]) and not np.isnan(slow_ema[i]):
            macd[i] = fast_ema[i] - slow_ema[i]

    # First stochastic of MACD
    stoch1 = _njit_stochastic_of_series(macd, cycle_length)

    # Smoothed stoch1 (EMA)
    stoch1_ema = _njit_ema_of_series(stoch1, cycle_length)

    # Second stochastic of smoothed stoch1
    stoch2 = _njit_stochastic_of_series(stoch1_ema, cycle_length)

    # Final smoothed stoch2 (EMA) = STC
    stc = _njit_ema_of_series(stoch2, cycle_length)

    return stc


@njit(cache=True)
def _njit_stochastic_of_series(
    values: np.ndarray,
    length: int,
) -> np.ndarray:
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(length - 1, n):
        low_val = np.inf
        high_val = -np.inf
        valid = True
        for j in range(i - length + 1, i + 1):
            if np.isnan(values[j]):
                valid = False
                break
            if values[j] < low_val:
                low_val = values[j]
            if values[j] > high_val:
                high_val = values[j]
        if valid:
            denom = high_val - low_val
            if denom > 1e-12:
                result[i] = ((values[i] - low_val) / denom) * 100.0
            else:
                result[i] = 50.0
    return result


@njit(cache=True)
def _njit_ema_of_series(
    values: np.ndarray,
    length: int,
) -> np.ndarray:
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    alpha = 2.0 / (length + 1.0)

    # Find first `length` valid values to seed the EMA
    valid_indices = []
    for i in range(n):
        if not np.isnan(values[i]):
            valid_indices.append(i)
            if len(valid_indices) == length:
                break

    if len(valid_indices) < length:
        return result

    # Seed EMA from first `length` valid values
    seed = 0.0
    for idx in valid_indices:
        seed += values[idx]
    seed_idx = valid_indices[-1]
    result[seed_idx] = seed / float(length)

    # Continue EMA from seed_idx+1 onward
    for i in range(seed_idx + 1, n):
        if np.isnan(values[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]

    return result


@handle_pandas_ta_errors
def schaff_trend_cycle(
    close: pd.Series,
    fast_length: int = 23,
    slow_length: int = 50,
    cycle_length: int = 10,
) -> pd.Series:
    """Schaff Trend Cycle (STC).

    2段階の確率的MACDベースのサイクル検出オシレーター。
    0-100のスケールでトレンドの方向と転換点を検出する。
    """
    if fast_length < 2:
        raise ValueError("fast_length must be >= 2")
    if slow_length < 2:
        raise ValueError("slow_length must be >= 2")
    if cycle_length < 2:
        raise ValueError("cycle_length must be >= 2")
    if fast_length >= slow_length:
        raise ValueError("fast_length must be < slow_length")

    validation = validate_series_params(
        close,
        slow_length,
        min_data_length=slow_length + cycle_length,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"STC_{fast_length}_{slow_length}_{cycle_length}",
        )

    result = _njit_stc_loop(
        close.to_numpy(dtype=float),
        fast_length,
        slow_length,
        cycle_length,
    )
    return pd.Series(
        result,
        index=close.index,
        name=f"STC_{fast_length}_{slow_length}_{cycle_length}",
    )
