"""Fibonacci Cycle Indicator."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit, prange

from ...data_validation import handle_pandas_ta_errors, validate_series_params


@njit(parallel=True, cache=True)
def _njit_fibonacci_cycle_loop(
    prices: np.ndarray,
    cycle_periods: np.ndarray,
    fib_ratios: np.ndarray,
    max_period: int,
) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan)
    n_cycles = len(cycle_periods)
    n_ratios = len(fib_ratios)
    inv_r_len = 1.0 / n_ratios

    for i in prange(max_period, n):
        sq_sum = 0.0
        pos_count = 0
        sign_sum = 0
        has_val = False

        for p_idx in range(n_cycles):
            period = cycle_periods[p_idx]
            if i >= period:
                p_start = i - period + 1
                p0 = prices[p_start]
                if p0 != 0:
                    ret = (prices[i] - p0) / p0
                    r_sum = 0.0
                    for r_idx in range(n_ratios):
                        r_sum += ret / fib_ratios[r_idx]

                    v = r_sum * inv_r_len
                    if v != 0:
                        sq_sum += 1.0 / abs(v)
                        pos_count += 1
                        if v > 0:
                            sign_sum += 1
                        else:
                            sign_sum -= 1
                        has_val = True

        if has_val and pos_count > 0:
            final_v = (pos_count / sq_sum) * (1.0 if sign_sum >= 0 else -1.0)
            result[i] = final_v

    for i in range(max_period + 1, n):
        if np.isfinite(result[i]) and np.isfinite(result[i - 1]):
            result[i] = 0.3 * result[i] + 0.7 * result[i - 1]

    return result


@handle_pandas_ta_errors
def fibonacci_cycle(
    close: pd.Series,
    cycle_periods: list[int] | None = None,
    fib_ratios: list[float] | None = None,
) -> Tuple[pd.Series, pd.Series]:
    """Fibonacci Cycle Indicator."""
    if cycle_periods is None:
        cycle_periods = [8, 13, 21, 34, 55]
    if fib_ratios is None:
        fib_ratios = [0.618, 1.0, 1.618, 2.618]

    if not cycle_periods:
        raise ValueError("cycle_periods must not be empty")
    if not fib_ratios:
        raise ValueError("fib_ratios must not be empty")

    validation = validate_series_params(
        close, max(cycle_periods), min_data_length=max(cycle_periods)
    )
    if validation is not None:
        nan_cycle = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"FIBO_CYCLE_{len(cycle_periods)}",
        )
        nan_sig = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"FIBO_SIGNAL_{len(cycle_periods)}",
        )
        return nan_cycle, nan_sig

    prices = close.astype(float).to_numpy()
    max_period = max(cycle_periods)

    c_p = np.array(cycle_periods, dtype=np.int64)
    f_r = np.array(fib_ratios, dtype=np.float64)

    result = _njit_fibonacci_cycle_loop(prices, c_p, f_r, max_period)

    fibonacci_cycle_result = pd.Series(
        result, index=close.index, name=f"FIBO_CYCLE_{len(cycle_periods)}"
    )
    signal = fibonacci_cycle_result.rolling(window=3).mean()
    signal.name = f"FIBO_SIGNAL_{len(cycle_periods)}"

    return fibonacci_cycle_result, signal