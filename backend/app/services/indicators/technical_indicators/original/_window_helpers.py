"""原始指標で共有する Numba 窓集計ヘルパー."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _window_sum(values: np.ndarray, start: int, end: int) -> float:
    total = 0.0
    for idx in range(start, end):
        total += values[idx]
    return total


@njit(cache=True)
def _window_mean(values: np.ndarray, start: int, end: int) -> float:
    length = end - start
    if length <= 0:
        return 0.0
    return _window_sum(values, start, end) / float(length)


@njit(cache=True)
def _window_mean_and_std(values: np.ndarray, start: int, end: int) -> tuple[float, float]:
    length = end - start
    if length <= 0:
        return 0.0, 0.0

    total = 0.0
    sq_total = 0.0
    for idx in range(start, end):
        val = values[idx]
        total += val
        sq_total += val * val

    mean = total / float(length)
    variance = (sq_total / float(length)) - (mean * mean)
    if variance < 0.0:
        variance = 0.0
    return mean, np.sqrt(variance)


@njit(cache=True)
def _window_mean_and_std_finite(
    values: np.ndarray, start: int, end: int
) -> tuple[float, float, int]:
    total = 0.0
    sq_total = 0.0
    count = 0

    for idx in range(start, end):
        val = values[idx]
        if np.isfinite(val):
            total += val
            sq_total += val * val
            count += 1

    if count == 0:
        return 0.0, 0.0, 0

    mean = total / float(count)
    variance = (sq_total / float(count)) - (mean * mean)
    if variance < 0.0:
        variance = 0.0
    return mean, np.sqrt(variance), count


@njit(cache=True)
def _window_min_max(values: np.ndarray, start: int, end: int) -> tuple[float, float]:
    min_val = values[start]
    max_val = values[start]
    for idx in range(start + 1, end):
        val = values[idx]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    return min_val, max_val


@njit(cache=True)
def _window_range(values: np.ndarray, start: int, end: int, scale: float) -> float:
    min_val, max_val = _window_min_max(values, start, end)
    return (max_val - min_val) / scale
