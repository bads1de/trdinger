"""フィルタ系の独自テクニカル指標."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numba import njit, prange

from ...data_validation import handle_pandas_ta_errors, validate_series_params

ALPHA_MIN: Final[float] = 0.01
ALPHA_MAX: Final[float] = 1.0


@njit(parallel=True, cache=True)
def _njit_frama_loop(
    prices: np.ndarray,
    length: int,
    half: int,
    log2: float,
    w: float,
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    n = len(prices)
    filt = np.full(n, np.nan)
    for i in prange(length - 1, n):
        n1_high = -1e12
        n1_low = 1e12
        for j in range(i - length + 1, i - half + 1):
            if prices[j] > n1_high:
                n1_high = prices[j]
            if prices[j] < n1_low:
                n1_low = prices[j]
        n1 = (n1_high - n1_low) / half

        n2_high = -1e12
        n2_low = 1e12
        for j in range(i - half + 1, i + 1):
            if prices[j] > n2_high:
                n2_high = prices[j]
            if prices[j] < n2_low:
                n2_low = prices[j]
        n2 = (n2_high - n2_low) / half

        n3_high = -1e12
        n3_low = 1e12
        for j in range(i - length + 1, i + 1):
            if prices[j] > n3_high:
                n3_high = prices[j]
            if prices[j] < n3_low:
                n3_low = prices[j]
        n3 = (n3_high - n3_low) / length

        if n1 > 1e-9 and n2 > 1e-9 and n3 > 1e-9:
            dimen = (np.log(n1 + n2) - np.log(n3)) / log2
        else:
            dimen = 1.0

        alpha = np.exp(w * (dimen - 1.0))
        if alpha < alpha_min:
            alpha = alpha_min
        if alpha > alpha_max:
            alpha = alpha_max

        if i == length - 1:
            filt[i] = prices[i]
        else:
            prev_filt = filt[i - 1]
            if np.isfinite(prev_filt):
                filt[i] = alpha * prices[i] + (1.0 - alpha) * prev_filt
            else:
                filt[i] = prices[i]
    return filt


@njit(cache=True)
def _njit_super_smoother_loop(
    prices: np.ndarray, c1: float, c2: float, c3: float
) -> np.ndarray:
    n = len(prices)
    filt = np.full(n, np.nan)
    filt[0] = prices[0]
    if n > 1:
        filt[1] = prices[1]
    for i in range(2, n):
        filt[i] = (
            c1 * (prices[i] + prices[i - 1]) / 2.0 + c2 * filt[i - 1] + c3 * filt[i - 2]
        )
    return filt


@njit(cache=True)
def _njit_mcginley_dynamic_loop(prices: np.ndarray, length: int, k: float) -> np.ndarray:
    n = len(prices)
    result = np.empty(n)
    result[:] = np.nan

    if n == 0:
        return result

    result[0] = prices[0]

    for i in range(1, n):
        price = prices[i]
        prev_md = result[i - 1]

        if np.isnan(prev_md) or prev_md == 0:
            result[i] = price
            continue

        ratio = price / prev_md
        if ratio < 0.1:
            ratio = 0.1
        elif ratio > 10.0:
            ratio = 10.0

        denominator = k * length * (ratio**4)
        if denominator < 1e-10:
            denominator = 1e-10

        md_change = (price - prev_md) / denominator
        result[i] = prev_md + md_change

    return result


@handle_pandas_ta_errors
def frama(close: pd.Series, length: int = 16, slow: int = 200) -> pd.Series:
    """Fractal Adaptive Moving Average (FRAMA)."""
    if length < 4:
        length = 4
    if length % 2 != 0:
        length += 1
    if slow < 1:
        slow = 1

    validation = validate_series_params(close, length, min_data_length=length)
    if validation is not None:
        return pd.Series(np.full(len(close), np.nan), index=close.index, name="FRAMA")

    prices = close.astype(float).to_numpy()
    half = length // 2
    log2 = np.log(2.0)
    slow_float = float(slow)
    w = 2.303 * np.log(2.0 / (slow_float + 1.0))

    result = _njit_frama_loop(
        prices,
        length,
        half,
        log2,
        w,
        ALPHA_MIN,
        ALPHA_MAX,
    )

    return pd.Series(result, index=close.index, name="FRAMA")


@handle_pandas_ta_errors
def super_smoother(close: pd.Series, length: int = 10) -> pd.Series:
    """Ehlers 2-Pole Super Smoother Filter."""
    if length < 2:
        raise ValueError("length must be >= 2")

    validation = validate_series_params(close, length, min_data_length=length)
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan), index=close.index, name="SUPER_SMOOTHER"
        )

    prices = close.astype(float).to_numpy()
    f = (np.sqrt(2.0) * np.pi) / float(length)
    a = np.exp(-f)
    c2 = 2.0 * a * np.cos(f)
    c3 = -(a**2)
    c1 = 1.0 - c2 - c3

    result = _njit_super_smoother_loop(prices, c1, c2, c3)
    return pd.Series(result, index=close.index, name="SUPER_SMOOTHER")


@handle_pandas_ta_errors
def mcginley_dynamic(close: pd.Series, length: int = 10, k: float = 0.6) -> pd.Series:
    """McGinley Dynamic (MD)."""
    if length < 1:
        raise ValueError("length must be >= 1")

    validation = validate_series_params(close, length, min_data_length=length)
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan), index=close.index, name=f"MCGINLEY_{length}"
        )

    if k <= 0:
        raise ValueError("k must be > 0")

    prices = close.astype(float).to_numpy()
    result = _njit_mcginley_dynamic_loop(prices, length, k)

    return pd.Series(result, index=close.index, name=f"MCGINLEY_{length}")


@handle_pandas_ta_errors
def calculate_mcginley_dynamic(data, length=10, k=0.6):
    """McGinley Dynamic計算のラッパーメソッド."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")

    required_columns = ["close"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    close = data["close"]
    md = mcginley_dynamic(close, length, k)

    result = pd.DataFrame(
        {
            md.name: md,
        },
        index=data.index,
    )

    return result
