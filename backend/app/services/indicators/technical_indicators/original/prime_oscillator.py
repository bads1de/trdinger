"""Prime Number Oscillator."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit, prange

from ._window_helpers import _window_sum
from ...data_validation import (
    create_nan_series_bundle,
    handle_pandas_ta_errors,
    validate_series_params,
)


@njit(cache=True)
def _njit_is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _get_prime_sequence(length: int) -> list[int]:
    primes: list[int] = []
    num = 2
    while len(primes) < length:
        if _njit_is_prime(num):
            primes.append(num)
        num += 1
    return primes


@njit(cache=True)
def _njit_prime_oscillator_loop(
    prices: np.ndarray, primes: np.ndarray, lookback_limit: int = 200
) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan)
    n_primes = len(primes)
    if n_primes == 0:
        return result

    max_p = 0
    for p in primes:
        if p > max_p:
            max_p = p

    if n < max_p:
        return result

    weights = 1.0 / primes.astype(np.float64)
    w_sum = 0.0
    for w in weights:
        w_sum += w

    bar_sums = np.zeros(n)
    bar_sq_sums = np.zeros(n)
    bar_counts = np.zeros(n, dtype=np.int32)
    unscaled_osc = np.zeros(n)

    for i in prange(n):
        s = 0.0
        sq_s = 0.0
        c = 0
        w_s = 0.0
        for p_idx in range(n_primes):
            p = primes[p_idx]
            if i >= p:
                p_prev = prices[i - p]
                if p_prev != 0:
                    chg = (prices[i] - p_prev) / p_prev
                    s += chg
                    sq_s += chg * chg
                    c += 1
                    w_s += weights[p_idx] * chg
        bar_sums[i] = s
        bar_sq_sums[i] = sq_s
        bar_counts[i] = c
        unscaled_osc[i] = w_s / w_sum

    for i in prange(max_p, n):
        lookback = min(lookback_limit, i)
        start_j = i - lookback + 1

        total_sum = _window_sum(bar_sums, start_j, i + 1)
        total_sq_sum = _window_sum(bar_sq_sums, start_j, i + 1)
        total_count = _window_sum(bar_counts, start_j, i + 1)

        if total_count > 0:
            m = total_sum / total_count
            v = (total_sq_sum / total_count) - (m * m)
            if v > 1e-12:
                result[i] = (unscaled_osc[i] / np.sqrt(v)) * 100.0
            else:
                result[i] = unscaled_osc[i]
        else:
            result[i] = unscaled_osc[i]

    return result


@handle_pandas_ta_errors
def prime_oscillator(
    close: pd.Series, length: int = 14, signal_length: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Prime Number Oscillator."""
    if length < 2:
        raise ValueError("length must be >= 2")

    validation = validate_series_params(close, length, min_data_length=length)
    if validation is not None:
        nan_osc = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"PRIME_OSC_{length}",
        )
        nan_sig = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"PRIME_SIGNAL_{length}_{signal_length}",
        )
        return nan_osc, nan_sig

    if signal_length < 2:
        raise ValueError("signal_length must be >= 2")

    prices = close.astype(float).to_numpy()
    result = np.empty_like(prices)
    result[:] = np.nan

    primes = _get_prime_sequence(length)
    from typing import cast
    if not primes:
        return cast(Tuple[pd.Series, pd.Series], create_nan_series_bundle(close, 2))

    primes_array = np.array(primes, dtype=np.int64)
    result = _njit_prime_oscillator_loop(prices, primes_array, 200)

    oscillator = pd.Series(result, index=close.index, name=f"PRIME_OSC_{length}")
    signal = oscillator.rolling(window=signal_length).mean()
    signal.name = f"PRIME_SIGNAL_{length}_{signal_length}"  # type: ignore[reportAttributeAccessIssue]

    return oscillator, signal