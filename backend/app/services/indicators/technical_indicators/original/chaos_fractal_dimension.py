"""Chaos Theory Fractal Dimension (CTFD)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit, prange

from ._window_helpers import _window_mean_and_std_finite
from ...data_validation import handle_pandas_ta_errors, validate_multi_series_params


@njit(cache=True)
def _calculate_correlation_dimension_impl(
    prices: np.ndarray, embedding_dim: int = 3, time_delay: int = 1
) -> float:
    n_prices = len(prices)
    if n_prices < embedding_dim * 2:
        return 1.0

    n_points = n_prices - (embedding_dim - 1) * time_delay
    if n_points <= 0:
        return 1.0

    embedded = np.zeros((n_points, embedding_dim))
    for i in range(embedding_dim):
        start_idx = i * time_delay
        embedded[:, i] = prices[start_idx : start_idx + n_points]

    n_dist = n_points * (n_points - 1) // 2
    if n_dist == 0:
        return 1.0

    distances = np.zeros(n_dist)
    count = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d_sq = 0.0
            for k in range(embedding_dim):
                diff = embedded[i, k] - embedded[j, k]
                d_sq += diff * diff
            distances[count] = np.sqrt(d_sq)
            count += 1

    distances.sort()

    if len(distances) > 10:
        mid_point = len(distances) // 2
        if mid_point > 1:
            sum_log_low = 0.0
            for i in range(1, mid_point):
                if distances[i] > 1e-12:
                    sum_log_low += np.log(distances[i])
            low_mean = sum_log_low / (mid_point - 1)

            sum_log_high = 0.0
            for i in range(mid_point, len(distances)):
                if distances[i] > 1e-12:
                    sum_log_high += np.log(distances[i])
            high_mean = sum_log_high / (len(distances) - mid_point)

            low_log_c = np.log(mid_point / len(distances))
            high_log_c = np.log(0.5)

            if abs(high_mean - low_mean) > 1e-12:
                dimension = (high_log_c - low_log_c) / (high_mean - low_mean)
                if dimension < 1.0:
                    dimension = 1.0
                if dimension > 5.0:
                    dimension = 5.0
                return dimension

    return 1.0


@njit(cache=True)
def _njit_solve_3x3(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    det = (
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
        + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    )

    if abs(det) < 1e-15:
        return np.zeros(3)

    inv_det = 1.0 / det

    x = (
        b[0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (b[1] * A[2, 2] - A[1, 2] * b[2])
        + A[0, 2] * (b[1] * A[2, 1] - A[1, 1] * b[2])
    ) * inv_det

    y = (
        A[0, 0] * (b[1] * A[2, 2] - A[1, 2] * b[2])
        - b[0] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
        + A[0, 2] * (A[1, 0] * b[2] - b[1] * A[2, 0])
    ) * inv_det

    z = (
        A[0, 0] * (A[1, 1] * b[2] - b[1] * A[2, 1])
        - A[0, 1] * (A[1, 0] * b[2] - b[1] * A[2, 0])
        + b[0] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    ) * inv_det

    return np.array([x, y, z])


@njit(parallel=True, cache=True)
def _njit_ctfd_raw_pass(
    prices: np.ndarray,
    volumes: np.ndarray,
    length: int,
    embedding_dim: int,
    min_period: int,
) -> np.ndarray:
    n = len(prices)
    raw_scores = np.full(n, np.nan)

    for i in prange(min_period, n):
        window_start = i - length + 1
        p_win = prices[window_start : i + 1]
        v_win = volumes[window_start : i + 1]

        p_len = len(p_win)
        p_chg = np.zeros(p_len)
        v_chg = np.zeros(p_len)
        for j in range(1, p_len):
            p_chg[j] = p_win[j] - p_win[j - 1]
            v_chg[j] = v_win[j] - v_win[j - 1]

        corr_dim = _calculate_correlation_dimension_impl(p_win, embedding_dim, 1)
        chaos_score = corr_dim

        if p_len > 5:
            sx, sx2, sx3, sx4 = 0.0, 0.0, 0.0, 0.0
            sy, sxy, sx2y = 0.0, 0.0, 0.0

            for j in range(p_len):
                px = p_chg[j]
                vx = v_chg[j]
                px2 = px * px
                sx += px
                sx2 += px2
                sx3 += px2 * px
                sx4 += px2 * px2
                sy += vx
                sxy += px * vx
                sx2y += px2 * vx

            A = np.array([[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, float(p_len)]])
            b = np.array([sx2y, sxy, sy])

            coeffs = _njit_solve_3x3(A, b)

            if np.any(coeffs != 0):
                resid_sum = 0.0
                resid_sq_sum = 0.0
                for j in range(p_len):
                    px = p_chg[j]
                    pred = coeffs[0] * px * px + coeffs[1] * px + coeffs[2]
                    res = v_chg[j] - pred
                    resid_sum += res
                    resid_sq_sum += res * res

                resid_mean = resid_sum / p_len
                nonlin_resid = np.sqrt(
                    max(0.0, (resid_sq_sum / p_len) - (resid_mean**2))
                )

                px_sum = sx
                px2_sum = sx2
                px4_sum = sx4
                px_px2_sum = sx3

                px_mean = px_sum / p_len
                px2_mean = px2_sum / p_len

                p_var = (px2_sum / p_len) - (px_mean**2)
                px2_var = (px4_sum / p_len) - (px2_mean**2)
                cov = (px_px2_sum / p_len) - (px_mean * px2_mean)

                lin_corr = 0.0
                if p_var > 1e-12 and px2_var > 1e-12:
                    lin_corr = cov / np.sqrt(p_var * px2_var)

                v_avg = sy / p_len
                v_ss = 0.0
                for j in range(p_len):
                    v_ss += (v_chg[j] - v_avg) ** 2
                vol_std = np.sqrt(v_ss / p_len)

                chaos_score = (
                    corr_dim * 0.4
                    + abs(lin_corr) * 0.3
                    + (nonlin_resid / (vol_std + 1e-6)) * 0.3
                )
            else:
                chaos_score = corr_dim * 0.7 + 0.3

        raw_scores[i] = 1.0 / (1.0 + chaos_score)

    return raw_scores


@njit(cache=True)
def _njit_ctfd_normalize(raw_scores: np.ndarray, min_period: int) -> np.ndarray:
    n = len(raw_scores)
    result = np.full(n, np.nan)

    for i in range(min_period, n):
        pred = raw_scores[i]
        lookback = min(200, i - min_period + 1)

        start_k = i - lookback + 1
        if start_k < 0:
            start_k = 0

        recent_mean, recent_std, recent_count = _window_mean_and_std_finite(
            result, start_k, i
        )

        if recent_count > 10:
            if recent_std > 1e-12:
                norm = (pred - recent_mean) / recent_std
                if norm < -1.0:
                    norm = -1.0
                if norm > 1.0:
                    norm = 1.0
                result[i] = norm
            else:
                result[i] = pred
        else:
            result[i] = pred

    return result


def _chaos_fractal_dimension_loop(
    prices: np.ndarray, volumes: np.ndarray, length: int, embedding_dim: int
) -> np.ndarray:
    min_period = max(length, 30)
    raw_scores = _njit_ctfd_raw_pass(prices, volumes, length, embedding_dim, min_period)
    return _njit_ctfd_normalize(raw_scores, min_period)


@handle_pandas_ta_errors
def chaos_fractal_dimension(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    length: int = 25,
    embedding_dim: int = 3,
    signal_length: int = 4,
) -> Tuple[pd.Series, pd.Series]:
    """Chaos Theory Fractal Dimension (CTFD)."""
    if length < 15:
        raise ValueError("length must be >= 15")
    if embedding_dim < 2 or embedding_dim > 5:
        raise ValueError("embedding_dim must be between 2 and 5")
    if signal_length < 2:
        raise ValueError("signal_length must be >= 2")

    validation = validate_multi_series_params(
        {"close": close, "high": high, "low": low, "volume": volume},
        length,
        min_data_length=length,
    )
    if validation is not None:
        nan_ctf = pd.Series(
            np.full(len(close), np.nan), index=close.index, name="CHAOS_FRACTAL_DIM"
        )
        nan_sig = pd.Series(
            np.full(len(close), np.nan), index=close.index, name="CTFD_SIGNAL"
        )
        return nan_ctf, nan_sig

    prices = close.astype(float).to_numpy()
    volumes = volume.astype(float).to_numpy()

    result = _chaos_fractal_dimension_loop(prices, volumes, length, embedding_dim)

    ctf_series = pd.Series(result, index=close.index, name="CHAOS_FRACTAL_DIM")
    signal = ctf_series.rolling(window=signal_length, min_periods=1).mean()
    signal.name = "CTFD_SIGNAL"  # type: ignore[reportAttributeAccessIssue]

    return ctf_series, signal