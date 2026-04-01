"""Quantum Flow Analysis."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit, prange

from ._window_helpers import _window_mean_and_std
from ...data_validation import handle_pandas_ta_errors, validate_multi_series_params


@njit(parallel=True, cache=True)
def _simple_wavelet_transform(data: np.ndarray, scale: int) -> np.ndarray:
    scale = int(scale)
    n = len(data)
    result = np.full(n, np.nan)
    if n < scale or scale < 2:
        return result

    half = scale // 2
    sqrt_scale = np.sqrt(float(scale))
    inv_half = 1.0 / half

    for i in prange(scale - 1, n):
        sum_first = 0.0
        sum_second = 0.0
        for j in range(half):
            sum_first += data[i - scale + 1 + j]
            sum_second += data[i - half + 1 + j]

        diff = (sum_second * inv_half) - (sum_first * inv_half)
        result[i] = diff * sqrt_scale

    return result


@njit(parallel=True, cache=True)
def _njit_quantum_flow_loop(
    prices,
    highs,
    lows,
    volumes,
    length,
    wavelet_result,
):
    n = len(prices)
    quantum_flow = np.zeros(n)

    price_change = np.zeros(n)
    volume_change = np.zeros(n)
    for i in prange(1, n):
        price_change[i] = prices[i] - prices[i - 1]
        volume_change[i] = volumes[i] - volumes[i - 1]

    correlation_score = np.zeros(n)
    for i in prange(length, n):
        sum_x = 0.0
        sum_y = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0
        sum_xy = 0.0

        for j in range(i - length + 1, i + 1):
            vx = price_change[j]
            vy = volume_change[j]
            sum_x += vx
            sum_y += vy
            sum_x2 += vx * vx
            sum_y2 += vy * vy
            sum_xy += vx * vy

        denom = np.sqrt(
            (length * sum_x2 - sum_x * sum_x) * (length * sum_y2 - sum_y * sum_y)
        )
        if denom > 1e-12:
            correlation_score[i] = (length * sum_xy - sum_x * sum_y) / denom
        else:
            correlation_score[i] = 0.0

    volatility = np.zeros(n)
    for i in prange(n):
        if prices[i] != 0:
            volatility[i] = (highs[i] - lows[i]) / prices[i]

    raw_integrated = np.zeros(n)
    for i in prange(length, n):
        wavelet_comp = wavelet_result[i] if np.isfinite(wavelet_result[i]) else 0.0
        raw_integrated[i] = (
            wavelet_comp * 0.4 + correlation_score[i] * 0.3 + volatility[i] * 0.3
        )

    for i in range(length, n):
        integrated = raw_integrated[i]
        lookback = min(200, i)
        if lookback >= length:
            _, std_val = _window_mean_and_std(raw_integrated, i - lookback + 1, i + 1)
            if std_val > 1e-12:
                integrated = integrated / std_val * 0.5
        quantum_flow[i] = integrated

    return quantum_flow


@handle_pandas_ta_errors
def quantum_flow(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    length: int = 14,
    flow_length: int = 9,
) -> Tuple[pd.Series, pd.Series]:
    """Quantum Flow Analysis."""
    length = int(length)
    flow_length = int(flow_length)
    if length < 5:
        raise ValueError("length must be >= 5")
    if flow_length < 3:
        raise ValueError("flow_length must be >= 3")

    validation = validate_multi_series_params(
        {"close": close, "high": high, "low": low, "volume": volume},
        length,
        min_data_length=length,
    )
    if validation is not None:
        nan_flow = pd.Series(
            np.full(len(close), np.nan), index=close.index, name="QUANTUM_FLOW"
        )
        nan_sig = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name="QUANTUM_FLOW_SIGNAL",
        )
        return nan_flow, nan_sig

    prices = close.astype(float).to_numpy()
    highs = high.astype(float).to_numpy()
    lows = low.astype(float).to_numpy()
    volumes = volume.astype(float).to_numpy()

    wavelet_result = _simple_wavelet_transform(prices, length)
    flow_values = _njit_quantum_flow_loop(
        prices, highs, lows, volumes, length, wavelet_result
    )

    signal = (
        pd.Series(flow_values, index=close.index).rolling(window=flow_length).mean()
    )
    signal.name = "QUANTUM_FLOW_SIGNAL"

    flow_series = pd.Series(flow_values, index=close.index, name="QUANTUM_FLOW")
    return flow_series, signal