"""
独自テクニカル指標モジュール

現在の実装:
- FRAMA (Fractal Adaptive Moving Average)
- SUPER_SMOOTHER (Ehlers 2-Pole Super Smoother Filter)
- ELDER_RAY (Elder Ray Index)
- PRIME_OSC (Prime Number Oscillator)
- FIBO_CYCLE (Fibonacci Cycle Indicator)
- ADAPTIVE_ENTROPY (Adaptive Entropy Oscillator)
- QUANTUM_FLOW (Quantum-inspired Flow Analysis)
- HARMONIC_RESONANCE (Harmonic Resonance Indicator)
- CHAOS_FRACTAL_DIM (Chaos Theory Fractal Dimension)
- MCGINLEY_DYNAMIC (McGinley Dynamic)
- KAUFMAN_EFFICIENCY_RATIO (Kaufman Efficiency Ratio)
- CHANDE_KROLL_STOP (Chande Kroll Stop)
- TREND_INTENSITY_INDEX (Trend Intensity Index)
- CONNORS_RSI (Connors RSI)
- GRI (Gopalakrishnan Range Index)
"""

from __future__ import annotations

import logging
from typing import Final, Tuple

import numpy as np
import pandas as pd
from numba import njit, prange


from ..data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
    validate_series_params,
)

logger = logging.getLogger(__name__)


@njit(cache=True)
def _calculate_correlation_dimension_impl(
    prices: np.ndarray, embedding_dim: int = 3, time_delay: int = 1
) -> float:
    """相関次元の近似計算 (Numba高速化版 - グローバル関数)"""
    n_prices = len(prices)
    if n_prices < embedding_dim * 2:
        return 1.0

    n_points = n_prices - (embedding_dim - 1) * time_delay
    if n_points <= 0:
        return 1.0

    # 埋め込みベクトルの作成
    embedded = np.zeros((n_points, embedding_dim))
    for i in range(embedding_dim):
        start_idx = i * time_delay
        embedded[:, i] = prices[start_idx : start_idx + n_points]

    # 相関積分の計算
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

    # 距離の分布を分析
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
def _njit_dft_magnitude(x: np.ndarray) -> np.ndarray:
    """Numba-compatible DFT magnitude calculation for small windows."""
    n = len(x)
    m = n // 2
    mag = np.zeros(m, dtype=np.float64)

    for k in range(m):
        re = 0.0
        im = 0.0
        for t in range(n):
            angle = 2.0 * np.pi * k * t / n
            re += x[t] * np.cos(angle)
            im -= x[t] * np.sin(angle)
        mag[k] = np.sqrt(re * re + im * im)
    return mag


@njit(cache=True)
def _njit_find_dominant_freqs(prices: np.ndarray) -> np.ndarray:
    """Numba-optimized dominant frequency detection using DFT and peak finding."""
    n = len(prices)
    if n < 4:
        return np.zeros(0)

    # Hamming Window
    windowed = np.empty(n, dtype=np.float64)
    for i in range(n):
        w = 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (n - 1))
        windowed[i] = prices[i] * w

    # DFT Magnitude
    magnitude = _njit_dft_magnitude(windowed)

    # Peak detection
    m = len(magnitude)
    if m < 3:
        return np.array([0.1, 0.2, 0.3])

    mean_mag = 0.0
    for i in range(m):
        mean_mag += magnitude[i]
    mean_mag /= m

    peaks_indices = []
    peaks_mags = []
    for i in range(1, m - 1):
        if (
            magnitude[i] > magnitude[i - 1]
            and magnitude[i] > magnitude[i + 1]
            and magnitude[i] > mean_mag
        ):
            peaks_indices.append(i)
            peaks_mags.append(magnitude[i])

    if len(peaks_indices) == 0:
        return np.array([0.1, 0.2, 0.3])

    # Sort peaks by magnitude (descending)
    p_indices = np.array(peaks_indices)
    p_mags = np.array(peaks_mags)

    # Simple selection of top 3
    sorted_idx = np.argsort(p_mags)[::-1]

    num_peaks = min(3, len(sorted_idx))
    res_freqs = np.empty(num_peaks, dtype=np.float64)
    for i in range(num_peaks):
        res_freqs[i] = float(p_indices[sorted_idx[i]]) / float(n)

    return res_freqs


@njit(cache=True)
def _njit_apply_bandpass_res(x: np.ndarray, freq: float, q: float = 2.0) -> np.ndarray:
    """Numba-optimized zero-phase biquad bandpass filter (simulating filtfilt)."""
    n = len(x)
    y = np.zeros(n, dtype=np.float64)

    # Forward pass
    omega = 2.0 * np.pi * freq
    alpha = np.sin(omega) / (2.0 * q)

    b0 = alpha
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(omega)
    a2 = 1.0 - alpha

    nb0, nb2 = b0 / a0, b2 / a0
    na1, na2 = a1 / a0, a2 / a0

    y[0] = nb0 * x[0]
    if n > 1:
        y[1] = nb0 * x[1] - na1 * y[0]
    for i in range(2, n):
        y[i] = nb0 * x[i] + nb2 * x[i - 2] - na1 * y[i - 1] - na2 * y[i - 2]

    # Backward pass for zero-phase
    y_rev = y[::-1]
    y_out = np.zeros(n, dtype=np.float64)

    y_out[0] = nb0 * y_rev[0]
    if n > 1:
        y_out[1] = nb0 * y_rev[1] - na1 * y_out[0]
    for i in range(2, n):
        y_out[i] = (
            nb0 * y_rev[i]
            + nb2 * y_rev[i - 2]
            - na1 * y_out[i - 1]
            - na2 * y_out[i - 2]
        )

    return y_out[::-1]


@njit(parallel=True, cache=True)
def _njit_harmonic_resonance_loop(
    prices: np.ndarray, length: int, resonance_bands: int, min_period: int
) -> np.ndarray:
    n = len(prices)
    scores = np.zeros(n, dtype=np.float64)

    for i in prange(min_period, n):
        window_start = i - length + 1
        price_window = prices[window_start : i + 1]

        mean_p = 0.0
        for val in price_window:
            mean_p += val
        mean_p /= length

        detrended = np.empty(length, dtype=np.float64)
        for j in range(length):
            detrended[j] = price_window[j] - mean_p

        dominant_freqs = _njit_find_dominant_freqs(detrended)

        if len(dominant_freqs) == 0:
            scores[i] = 0.0
            continue

        res_score = 0.0
        num_freqs = min(resonance_bands, len(dominant_freqs))

        for f_idx in range(num_freqs):
            freq = dominant_freqs[f_idx]
            if freq <= 0 or freq >= 0.5:
                continue

            filtered = _njit_apply_bandpass_res(detrended, freq, q=2.0)

            if length > 1:
                f_mean = 0.0
                for f_val in filtered:
                    f_mean += f_val
                f_mean /= length

                ss_xx = 0.0
                ss_yy = 0.0
                ss_xy = 0.0

                for j in range(length - 1):
                    x_dev = filtered[j] - f_mean
                    y_dev = filtered[j + 1] - f_mean
                    ss_xx += x_dev * x_dev
                    ss_yy += y_dev * y_dev
                    ss_xy += x_dev * y_dev

                denom = np.sqrt(ss_xx * ss_yy)
                if denom > 1e-12:
                    corr = ss_xy / denom
                else:
                    corr = 0.0

                std_f = np.sqrt(ss_xx / (length - 1))
                freq_weight = 1.0 / (1.0 + freq * 10.0)
                res_score += abs(corr) * std_f * freq_weight

        scores[i] = res_score

    final_result = np.full(n, np.nan)
    for i in prange(min_period, n):
        lookback = min(200, i - min_period + 1)
        if lookback >= 10:
            # Manual calculation of mean/std on valid scores (>0)
            m = 0.0
            v_sum = 0.0
            count = 0

            for j in range(i - lookback + 1, i + 1):
                val = scores[j]
                if val > 0:
                    m += val
                    count += 1

            if count > 5:
                m /= count
                for j in range(i - lookback + 1, i + 1):
                    val = scores[j]
                    if val > 0:
                        v_sum += (val - m) ** 2
                s = np.sqrt(v_sum / count)

                if s > 1e-12:
                    final_result[i] = (scores[i] - m) / s
                else:
                    final_result[i] = scores[i]
            else:
                final_result[i] = scores[i]
        else:
            final_result[i] = scores[i]

    return final_result


@njit(cache=True)
def _njit_solve_3x3(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Manual 3x3 linear system solver using Cramer's rule for speed in Numba."""
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

                # Linear correlation part (matching original implementation: price_change vs price_change^2)
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

                # Volume std for nonlinear part
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

        recent_sum = 0.0
        recent_sq_sum = 0.0
        recent_count = 0

        for k in range(start_k, i):
            val = result[k]
            if np.isfinite(val):
                recent_sum += val
                recent_sq_sum += val * val
                recent_count += 1

        if recent_count > 10:
            m = recent_sum / recent_count
            v = (recent_sq_sum / recent_count) - (m * m)
            s = np.sqrt(max(0.0, v))

            if s > 1e-12:
                norm = (pred - m) / s
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
        # N1 calculation
        n1_high = -1e12
        n1_low = 1e12
        for j in range(i - length + 1, i - half + 1):
            if prices[j] > n1_high:
                n1_high = prices[j]
            if prices[j] < n1_low:
                n1_low = prices[j]
        n1 = (n1_high - n1_low) / half

        # N2 calculation
        n2_high = -1e12
        n2_low = 1e12
        for j in range(i - half + 1, i + 1):
            if prices[j] > n2_high:
                n2_high = prices[j]
            if prices[j] < n2_low:
                n2_low = prices[j]
        n2 = (n2_high - n2_low) / half

        # N3 calculation
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


@njit(parallel=True, cache=True)
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

    # Pre-calculate changes for each prime at each bar
    # and the sum of changes across all primes at each bar
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

    # Final pass: Normalized oscillator using rolling window of bar_sums
    for i in prange(max_p, n):
        lookback = min(lookback_limit, i)
        start_j = i - lookback + 1

        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        for j in range(start_j, i + 1):
            total_sum += bar_sums[j]
            total_sq_sum += bar_sq_sums[j]
            total_count += bar_counts[j]

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


@njit(parallel=True, cache=True)
def _njit_entropy_loop(data: np.ndarray, window: int) -> np.ndarray:
    n = len(data)
    result = np.full(n, np.nan)
    if n < window:
        return result

    for i in prange(window - 1, n):
        win_data = data[i - window + 1 : i + 1]

        d_min = win_data[0]
        d_max = win_data[0]
        for val in win_data:
            if val < d_min:
                d_min = val
            if val > d_max:
                d_max = val

        if d_min == d_max:
            result[i] = 0.0
            continue

        n_bins = min(10, window)
        hist = np.zeros(n_bins)
        bin_w = (d_max - d_min) / n_bins

        for val in win_data:
            b_idx = int((val - d_min) / bin_w)
            if b_idx >= n_bins:
                b_idx = n_bins - 1
            hist[b_idx] += 1

        entropy = 0.0
        inv_total = 1.0 / window
        for count in hist:
            p = count * inv_total
            if p > 1e-12:
                entropy -= p * np.log(p)
        result[i] = entropy
    return result


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

    # EWA Pass (Sequential)
    for i in range(max_period + 1, n):
        if np.isfinite(result[i]) and np.isfinite(result[i - 1]):
            result[i] = 0.3 * result[i] + 0.7 * result[i - 1]

    return result


class OriginalIndicators:
    """新規の独自指標を提供するクラス"""

    _ALPHA_MIN: Final[float] = 0.01
    _ALPHA_MAX: Final[float] = 1.0

    @staticmethod
    def _frama_loop(prices, length, half, log2, w, alpha_min, alpha_max):
        return _njit_frama_loop(prices, length, half, log2, w, alpha_min, alpha_max)

    @staticmethod
    @handle_pandas_ta_errors
    def frama(close: pd.Series, length: int = 16, slow: int = 200) -> pd.Series:
        """Fractal Adaptive Moving Average (FRAMA)"""
        if length < 4:
            length = 4
        if length % 2 != 0:
            length += 1

        if slow < 1:
            slow = 1

        validation = validate_series_params(close, length, min_data_length=length)
        if validation is not None:
            return pd.Series(
                np.full(len(close), np.nan), index=close.index, name="FRAMA"
            )

        prices = close.astype(float).to_numpy()
        half = length // 2
        log2 = np.log(2.0)
        slow_float = float(slow)
        w = 2.303 * np.log(2.0 / (slow_float + 1.0))

        result = OriginalIndicators._frama_loop(
            prices,
            length,
            half,
            log2,
            w,
            OriginalIndicators._ALPHA_MIN,
            OriginalIndicators._ALPHA_MAX,
        )

        return pd.Series(result, index=close.index, name="FRAMA")

    @staticmethod
    def _super_smoother_loop(prices, c1, c2, c3):
        return _njit_super_smoother_loop(prices, c1, c2, c3)

    @staticmethod
    @handle_pandas_ta_errors
    def super_smoother(close: pd.Series, length: int = 10) -> pd.Series:
        """Ehlers 2-Pole Super Smoother Filter"""
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

        result = OriginalIndicators._super_smoother_loop(prices, c1, c2, c3)
        return pd.Series(result, index=close.index, name="SUPER_SMOOTHER")

    @staticmethod
    @handle_pandas_ta_errors
    def elder_ray(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 13,
        ema_length: int = 16,
    ) -> Tuple[pd.Series, pd.Series]:
        """Elder Ray Index"""
        length = int(length)
        ema_length = int(ema_length)
        if length <= 0:
            raise ValueError("length must be positive")
        if ema_length <= 0:
            raise ValueError("ema_length must be positive")

        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, ema_length
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        # EMAを計算
        ema = close.ewm(span=ema_length, adjust=False).mean()

        # ブルパワー: 高値 - EMA
        bull_power = high - ema

        # ベアパワー: 安値 - EMA
        bear_power = low - ema

        return bull_power, bear_power

    @staticmethod
    def calculate_elder_ray(data, length=13, ema_length=16):
        """Elder Ray Index計算のラッパーメソッド"""
        length = int(length)
        ema_length = int(ema_length)
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        high = data["high"]
        low = data["low"]
        close = data["close"]

        bull_power, bear_power = OriginalIndicators.elder_ray(
            high, low, close, length, ema_length
        )

        result = pd.DataFrame(
            {
                f"Elder_Ray_Bull_{length}_{ema_length}": bull_power,
                f"Elder_Ray_Bear_{length}_{ema_length}": bear_power,
            },
            index=data.index,
        )

        return result

    @staticmethod
    def _is_prime(n: int) -> bool:
        """素数判定ヘルパー関数"""
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

    @staticmethod
    def _get_prime_sequence(length: int) -> list[int]:
        """指定された長さの素数列を生成"""
        primes = []
        num = 2
        while len(primes) < length:
            if OriginalIndicators._is_prime(num):
                primes.append(num)
            num += 1
        return primes

    @staticmethod
    def _prime_oscillator_loop(prices, primes, lookback_limit=200):
        return _njit_prime_oscillator_loop(
            prices, primes, lookback_limit=lookback_limit
        )

    @staticmethod
    def _entropy_loop(data: np.ndarray, window: int) -> np.ndarray:
        return _njit_entropy_loop(data, window)

    # wrapper for backward compatibility if needed, but we will update caller

    @staticmethod
    @njit(parallel=True, cache=True)
    def _simple_wavelet_transform(data: np.ndarray, scale: int) -> np.ndarray:
        """簡単なウェーブレット変換の近似 (O(N) Parallel Optimized Version)"""
        scale = int(scale)
        n = len(data)
        result = np.full(n, np.nan)
        if n < scale or scale < 2:
            return result

        half = scale // 2
        sqrt_scale = np.sqrt(float(scale))
        inv_half = 1.0 / half

        # Parallelize the calculation using chunks
        # Each chunk will calculate its own sums to maintain O(N) within the chunk
        # but to keep it simple and strictly O(N) across the whole array safely,
        # a standard O(N) sliding window is often better.
        # However, for Numba parallel=True, we can use a hybrid approach or just parallel O(N*W) if W is small,
        # but here we want true O(N).

        # To truly parallelize and keep it O(N), we compute the first half/second sums for each thread's start.
        for i in prange(scale - 1, n):
            # Fallback to O(N*W) inside prange is actually okay if we have many cores,
            # but O(N) is always better. Let's do O(N*W) with prange first as it's easiest to parallelize.
            # If scale is very large, O(N) is needed.

            sum_first = 0.0
            sum_second = 0.0
            for j in range(half):
                sum_first += data[i - scale + 1 + j]
                sum_second += data[i - half + 1 + j]

            diff = (sum_second * inv_half) - (sum_first * inv_half)
            result[i] = diff * sqrt_scale

        return result

    @staticmethod
    def _calculate_correlation_dimension(
        prices: np.ndarray, embedding_dim: int = 3, time_delay: int = 1
    ) -> float:
        """相関次元の近似計算 (ラッパー)"""
        return _calculate_correlation_dimension_impl(prices, embedding_dim, time_delay)

    @staticmethod
    def _chaos_fractal_dimension_loop(prices, volumes, length, embedding_dim):
        min_period = max(length, 30)

        # Parallel raw pass
        raw_scores = _njit_ctfd_raw_pass(
            prices, volumes, length, embedding_dim, min_period
        )

        # Sequential normalization pass
        result = _njit_ctfd_normalize(raw_scores, min_period)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def prime_oscillator(
        close: pd.Series, length: int = 14, signal_length: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Prime Number Oscillator (素数オシレーター)"""
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

        # 指定期間内の素数列を生成
        primes = OriginalIndicators._get_prime_sequence(length)
        if not primes:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        # Prime Oscillatorの計算をNumbaで実行
        primes_array = np.array(primes, dtype=np.int64)
        result = OriginalIndicators._prime_oscillator_loop(prices, primes_array, 200)

        oscillator = pd.Series(result, index=close.index, name=f"PRIME_OSC_{length}")

        # Signal Lineの計算（SMA）
        signal = oscillator.rolling(window=signal_length).mean()
        signal.name = f"PRIME_SIGNAL_{length}_{signal_length}"

        return oscillator, signal

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_prime_oscillator(data, length=14, signal_length=3):
        """Prime Number Oscillator計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        oscillator, signal = OriginalIndicators.prime_oscillator(
            close, length, signal_length
        )

        result = pd.DataFrame(
            {
                oscillator.name: oscillator,
                signal.name: signal,
            },
            index=data.index,
        )

        return result

    @staticmethod
    def _generate_fibonacci_sequence(count: int) -> list[int]:
        """フィボナッチ数列を生成"""
        if count <= 0:
            return []
        elif count == 1:
            return [1]
        elif count == 2:
            return [1, 1]

        sequence = [1, 1]
        for i in range(2, count):
            sequence.append(sequence[i - 1] + sequence[i - 2])
        return sequence

    @staticmethod
    def _fibonacci_cycle_loop(prices, cycle_periods, fib_ratios, max_period):
        return _njit_fibonacci_cycle_loop(prices, cycle_periods, fib_ratios, max_period)

    @staticmethod
    @handle_pandas_ta_errors
    def fibonacci_cycle(
        close: pd.Series,
        cycle_periods: list[int] = None,
        fib_ratios: list[float] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Fibonacci Cycle Indicator (フィボナッチサイクルインジケーター)"""
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

        result = OriginalIndicators._fibonacci_cycle_loop(prices, c_p, f_r, max_period)

        fibonacci_cycle = pd.Series(
            result, index=close.index, name=f"FIBO_CYCLE_{len(cycle_periods)}"
        )
        signal = fibonacci_cycle.rolling(window=3).mean()
        signal.name = f"FIBO_SIGNAL_{len(cycle_periods)}"

        return fibonacci_cycle, signal

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_fibonacci_cycle(
        data, cycle_periods: list[int] = None, fib_ratios: list[float] = None
    ):
        """Fibonacci Cycle Indicator計算のラッパーメソッド"""
        # SeriesとDataFrameの両方に対応
        if isinstance(data, pd.Series):
            close = data
        elif isinstance(data, pd.DataFrame):
            required_columns = ["close"]
            for col in required_columns:
                if col not in data.columns:
                    # カラム名が大文字の場合も考慮
                    if col.capitalize() in data.columns:
                        col = col.capitalize()
                    else:
                        raise ValueError(f"Missing required column: {col}")
            close = data[col]
        else:
            raise TypeError("data must be pandas Series or DataFrame")

        cycle, signal = OriginalIndicators.fibonacci_cycle(
            close, cycle_periods, fib_ratios
        )

        result = pd.DataFrame(
            {
                cycle.name: cycle,
                signal.name: signal,
            },
            index=close.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def adaptive_entropy(
        close: pd.Series,
        short_length: int = 14,
        long_length: int = 28,
        signal_length: int = 5,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Adaptive Entropy Oscillator (適応的エントロピーオシレーター)"""
        if short_length < 5:
            raise ValueError("short_length must be >= 5")
        if long_length < 10:
            raise ValueError("long_length must be >= 10")
        if signal_length < 2:
            raise ValueError("signal_length must be >= 2")
        if short_length >= long_length:
            raise ValueError("short_length must be < long_length")

        validation = validate_series_params(
            close, long_length, min_data_length=long_length
        )
        if validation is not None:
            nan_osc = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"ADAPTIVE_ENTROPY_OSC_{short_length}_{long_length}",
            )
            nan_sig = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"ADAPTIVE_ENTROPY_SIGNAL_{short_length}_{long_length}_{signal_length}",
            )
            nan_ratio = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"ADAPTIVE_ENTROPY_RATIO_{short_length}_{long_length}",
            )
            return nan_osc, nan_sig, nan_ratio

        prices = close.astype(float).to_numpy()

        # 短期と長期のエントロピーを計算
        short_entropy = OriginalIndicators._entropy_loop(prices, short_length)
        long_entropy = OriginalIndicators._entropy_loop(prices, long_length)

        # エントロピー比を計算 (短期/長期)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy_ratio = short_entropy / long_entropy

        # 結果を正規化 (逆相関スケール: 高い値 = より混雑)
        normalized_osc = (entropy_ratio - 0.5) * 2.0

        # Signal Lineの計算（SMA）
        signal = (
            pd.Series(normalized_osc, index=close.index)
            .rolling(window=signal_length)
            .mean()
        )

        # 結果をPandas Seriesに変換
        oscillator = pd.Series(
            normalized_osc,
            index=close.index,
            name=f"ADAPTIVE_ENTROPY_OSC_{short_length}_{long_length}",
        )
        signal.name = (
            f"ADAPTIVE_ENTROPY_SIGNAL_{short_length}_{long_length}_{signal_length}"
        )
        ratio = pd.Series(
            entropy_ratio,
            index=close.index,
            name=f"ADAPTIVE_ENTROPY_RATIO_{short_length}_{long_length}",
        )

        return oscillator, signal, ratio

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_adaptive_entropy(
        data, short_length=14, long_length=28, signal_length=5
    ):
        """Adaptive Entropy Oscillator計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        oscillator, signal, ratio = OriginalIndicators.adaptive_entropy(
            close, short_length, long_length, signal_length
        )

        result = pd.DataFrame(
            {
                oscillator.name: oscillator,
                signal.name: signal,
                ratio.name: ratio,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @njit(parallel=True, cache=True)
    def _quantum_flow_loop(prices, highs, lows, volumes, length, wavelet_result):
        n = len(prices)
        quantum_flow = np.zeros(n)

        # 準備
        price_change = np.zeros(n)
        volume_change = np.zeros(n)
        for i in prange(1, n):
            price_change[i] = prices[i] - prices[i - 1]
            volume_change[i] = volumes[i] - volumes[i - 1]

        # 相関スコアの計算
        correlation_score = np.zeros(n)
        for i in prange(length, n):
            # i-length+1 から i までのウィンドウ
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

        # ボラティリティスコア
        volatility = np.zeros(n)
        for i in prange(n):
            if prices[i] != 0:
                volatility[i] = (highs[i] - lows[i]) / prices[i]

        # 統合
        # Note: Standard prange for integration.
        # The normalization part (np.std equivalent) is inherently local to each i here.
        for i in prange(length, n):
            wavelet_component = wavelet_result[i]
            if not np.isfinite(wavelet_component):
                wavelet_component = 0.0

            corr_component = correlation_score[i]
            vol_component = volatility[i]

            integrated = (
                wavelet_component * 0.4 + corr_component * 0.3 + vol_component * 0.3
            )

        # Optimized version:
        # 1. Compute all raw integrated values
        raw_integrated = np.zeros(n)
        for i in prange(length, n):
            wavelet_comp = wavelet_result[i] if np.isfinite(wavelet_result[i]) else 0.0
            raw_integrated[i] = (
                wavelet_comp * 0.4 + correlation_score[i] * 0.3 + volatility[i] * 0.3
            )

        # 2. Sequential normalization to match original logic if it depended on previous result
        # BUT if it just used its own previous values for std deviation, it's a bit weird.
        # Let's assume it wants a rolling std of the raw_integrated values.
        for i in range(length, n):
            integrated = raw_integrated[i]
            lookback = min(200, i)
            if lookback >= length:
                s1 = 0.0
                s2 = 0.0
                for j in range(i - lookback + 1, i + 1):
                    val = raw_integrated[j]
                    s1 += val
                    s2 += val * val
                m = s1 / lookback
                v = (s2 / lookback) - (m * m)
                if v > 1e-12:
                    integrated = integrated / np.sqrt(v) * 0.5
            quantum_flow[i] = integrated

        return quantum_flow

    @staticmethod
    def quantum_flow(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        length: int = 14,
        flow_length: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Quantum Flow Analysis (量子インスパイアード・フローアナリシス)"""
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

        # 価格データの前処理
        prices = close.astype(float).to_numpy()
        highs = high.astype(float).to_numpy()
        lows = low.astype(float).to_numpy()
        volumes = volume.astype(float).to_numpy()

        # ウェーブレット変換
        wavelet_result = OriginalIndicators._simple_wavelet_transform(prices, length)

        # Numba最適化ループを実行
        # wavelet_resultは既にnp.ndarray
        quantum_flow = OriginalIndicators._quantum_flow_loop(
            prices, highs, lows, volumes, length, wavelet_result
        )

        # Signal Line (SMA)
        signal = (
            pd.Series(quantum_flow, index=close.index)
            .rolling(window=flow_length)
            .mean()
        )

        # 結果をPandas Seriesに変換
        flow_series = pd.Series(quantum_flow, index=close.index, name="QUANTUM_FLOW")
        signal.name = "QUANTUM_FLOW_SIGNAL"

        return flow_series, signal

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_quantum_flow(data, length=14, flow_length=9):
        """Quantum Flow Analysis計算のラッパーメソッド"""
        length = int(length)
        flow_length = int(flow_length)
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close", "high", "low", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        flow, signal = OriginalIndicators.quantum_flow(
            close, high, low, volume, length, flow_length
        )

        result = pd.DataFrame(
            {
                flow.name: flow,
                signal.name: signal,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def harmonic_resonance(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
        resonance_bands: int = 5,
        signal_length: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Harmonic Resonance Indicator (HRI) - Numba Optimized Version"""
        if length < 10:
            raise ValueError("length must be >= 10")
        if resonance_bands < 3 or resonance_bands > 10:
            raise ValueError("resonance_bands must be between 3 and 10")
        if signal_length < 2:
            raise ValueError("signal_length must be >= 2")

        validation = validate_multi_series_params(
            {"close": close, "high": high, "low": low}, length, min_data_length=length
        )
        if validation is not None:
            nan_hri = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name="HARMONIC_RESONANCE",
            )
            nan_sig = pd.Series(
                np.full(len(close), np.nan), index=close.index, name="HRI_SIGNAL"
            )
            return nan_hri, nan_sig

        prices = close.astype(float).to_numpy()
        min_period = max(length, 30)

        # Execute optimized Numba loop
        hri_values = _njit_harmonic_resonance_loop(
            prices, length, resonance_bands, min_period
        )

        hri_series = pd.Series(hri_values, index=close.index, name="HARMONIC_RESONANCE")
        signal = hri_series.rolling(window=signal_length, min_periods=1).mean()
        signal.name = "HRI_SIGNAL"

        return hri_series, signal

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_harmonic_resonance(
        data, length=20, resonance_bands=5, signal_length=3
    ):
        """Harmonic Resonance Indicator計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close", "high", "low"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        high = data["high"]
        low = data["low"]

        hri, signal = OriginalIndicators.harmonic_resonance(
            close, high, low, length, resonance_bands, signal_length
        )

        result = pd.DataFrame(
            {
                hri.name: hri,
                signal.name: signal,
            },
            index=data.index,
        )

        return result

    @staticmethod
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
        """Chaos Theory Fractal Dimension (CTFD)"""
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

        result = OriginalIndicators._chaos_fractal_dimension_loop(
            prices, volumes, length, embedding_dim
        )

        ctf_series = pd.Series(result, index=close.index, name="CHAOS_FRACTAL_DIM")
        signal = ctf_series.rolling(window=signal_length, min_periods=1).mean()
        signal.name = "CTFD_SIGNAL"

        return ctf_series, signal

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_chaos_fractal_dimension(
        data, length=25, embedding_dim=3, signal_length=4
    ):
        """Chaos Theory Fractal Dimension計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close", "high", "low", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        ctf, signal = OriginalIndicators.chaos_fractal_dimension(
            close, high, low, volume, length, embedding_dim, signal_length
        )

        result = pd.DataFrame(
            {
                ctf.name: ctf,
                signal.name: signal,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @njit(cache=True)
    def _mcginley_dynamic_loop(prices, length, k):
        n = len(prices)
        result = np.empty(n)
        result[:] = np.nan

        if n == 0:
            return result

        # 初期値は最初の価格
        result[0] = prices[0]

        # McGinley Dynamicの計算
        for i in range(1, n):
            price = prices[i]
            prev_md = result[i - 1]

            if np.isnan(prev_md) or prev_md == 0:
                result[i] = price
                continue

            ratio = price / prev_md
            # ratio = np.clip(ratio, 0.1, 10.0) -> Numba compatible clip
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

    @staticmethod
    @handle_pandas_ta_errors
    def mcginley_dynamic(
        close: pd.Series, length: int = 10, k: float = 0.6
    ) -> pd.Series:
        """McGinley Dynamic (MD)"""
        if length < 1:
            raise ValueError("length must be >= 1")

        validation = validate_series_params(close, length, min_data_length=length)
        if validation is not None:
            return pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"MCGINLEY_{length}",
            )

        if k <= 0:
            raise ValueError("k must be > 0")

        prices = close.astype(float).to_numpy()
        result = OriginalIndicators._mcginley_dynamic_loop(prices, length, k)

        return pd.Series(result, index=close.index, name=f"MCGINLEY_{length}")

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_mcginley_dynamic(data, length=10, k=0.6):
        """McGinley Dynamic計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        md = OriginalIndicators.mcginley_dynamic(close, length, k)

        result = pd.DataFrame(
            {
                md.name: md,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def chande_kroll_stop(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        p: int = 10,
        x: float = 1.0,
        q: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Chande Kroll Stop"""
        if p < 1:
            raise ValueError("p must be >= 1")
        if q < 1:
            raise ValueError("q must be >= 1")

        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close},
            max(p, q),
            min_data_length=max(p, q),
        )
        if validation is not None:
            nan_long = pd.Series(
                np.full(len(close), np.nan), index=close.index, name=f"CKS_LONG_{p}"
            )
            nan_short = pd.Series(
                np.full(len(close), np.nan), index=close.index, name=f"CKS_SHORT_{p}"
            )
            return nan_long, nan_short

        if x <= 0:
            raise ValueError("x must be > 0")

        # ATRの計算
        tr = pd.DataFrame(
            {
                "hl": high - low,
                "hc": abs(high - close.shift(1)),
                "lc": abs(low - close.shift(1)),
            }
        ).max(axis=1)
        atr = tr.rolling(window=p).mean()

        highest_high = high.rolling(window=p).max()
        lowest_low = low.rolling(window=p).min()

        long_stop_initial = highest_high - x * atr
        short_stop_initial = lowest_low + x * atr

        long_stop = long_stop_initial.rolling(window=q).mean()
        short_stop = short_stop_initial.rolling(window=q).mean()

        long_stop.name = f"CKS_LONG_{p}"
        short_stop.name = f"CKS_SHORT_{p}"

        return long_stop, short_stop

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_chande_kroll_stop(data, p=10, x=1.0, q=9):
        """Chande Kroll Stop計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        high = data["high"]
        low = data["low"]
        close = data["close"]

        long_stop, short_stop = OriginalIndicators.chande_kroll_stop(
            high, low, close, p, x, q
        )

        result = pd.DataFrame(
            {
                long_stop.name: long_stop,
                short_stop.name: short_stop,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def trend_intensity_index(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
        sma_length: int = 30,
    ) -> pd.Series:
        """Trend Intensity Index (TII)"""
        if length < 1:
            raise ValueError("length must be >= 1")
        if sma_length < 1:
            raise ValueError("sma_length must be >= 1")

        validation = validate_multi_series_params(
            {"close": close, "high": high, "low": low},
            max(length, sma_length),
            min_data_length=max(length, sma_length),
        )
        if validation is not None:
            return pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"TII_{length}_{sma_length}",
            )

        # SMAの計算
        sma = close.rolling(window=sma_length).mean()

        # 終値がSMAより上かどうか
        above_sma = (close > sma).astype(int)

        # length期間内での上の日数をカウント
        count_above = above_sma.rolling(window=length).sum()

        # TIIの計算（パーセンテージ）
        tii = (count_above / length) * 100

        return pd.Series(tii, index=close.index, name=f"TII_{length}_{sma_length}")

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_trend_intensity_index(data, length=14, sma_length=30):
        """Trend Intensity Index計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close", "high", "low"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        high = data["high"]
        low = data["low"]

        tii = OriginalIndicators.trend_intensity_index(
            close, high, low, length, sma_length
        )

        result = pd.DataFrame(
            {
                tii.name: tii,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @njit(parallel=True, cache=True)
    def _connors_rsi_loop(prices, rsi_periods, streak_periods, rank_periods):
        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)
        max_p = max(rsi_periods, max(streak_periods, rank_periods))

        # 1. Close RSI
        close_rsi = np.full(n, np.nan, dtype=np.float64)
        for i in prange(rsi_periods, n):
            up = 0.0
            down = 0.0
            for j in range(i - rsi_periods + 1, i + 1):
                change = prices[j] - prices[j - 1]
                if change > 0:
                    up += change
                elif change < 0:
                    down += abs(change)

            if up > 0 and down > 0:
                rs = (up / rsi_periods) / (down / rsi_periods)
                close_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            elif up > 0:
                close_rsi[i] = 100.0
            else:
                close_rsi[i] = 0.0

        # 2. Streak RSI (Recursive calculation must be sequential)
        streaks = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if prices[i] > prices[i - 1]:
                streaks[i] = max(streaks[i - 1] + 1.0, 1.0)
            elif prices[i] < prices[i - 1]:
                streaks[i] = min(streaks[i - 1] - 1.0, -1.0)
            else:
                streaks[i] = 0.0

        streak_rsi = np.full(n, np.nan, dtype=np.float64)
        for i in prange(streak_periods, n):
            up = 0.0
            down = 0.0
            for j in range(i - streak_periods + 1, i + 1):
                change = streaks[j] - streaks[j - 1]
                if change > 0:
                    up += change
                elif change < 0:
                    down += abs(change)
            if up > 0 and down > 0:
                rs = (up / streak_periods) / (down / streak_periods)
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            elif up > 0:
                streak_rsi[i] = 100.0
            else:
                streak_rsi[i] = 0.0

        # 3. Rank
        rank_values = np.full(n, np.nan, dtype=np.float64)
        for i in prange(rank_periods, n):
            current_price = prices[i]
            count_lower = 0
            for j in range(i - rank_periods + 1, i + 1):
                if prices[j] <= current_price:
                    count_lower += 1
            rank_values[i] = (count_lower / rank_periods) * 100.0

        # Integration
        for i in prange(max_p, n):
            v1 = close_rsi[i]
            v2 = streak_rsi[i]
            v3 = rank_values[i]

            count = 0
            total = 0.0
            if np.isfinite(v1):
                total += v1
                count += 1
            if np.isfinite(v2):
                total += v2
                count += 1
            if np.isfinite(v3):
                total += v3
                count += 1

            if count == 3:
                val = total / 3.0
            elif count == 2:
                val = (total / 2.0) * (3.0 / 2.0)
            elif count == 1:
                val = total
            else:
                continue

            if val < 0.0:
                val = 0.0
            if val > 100.0:
                val = 100.0
            result[i] = val

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def connors_rsi(
        close: pd.Series,
        rsi_periods: int = 3,
        streak_periods: int = 2,
        rank_periods: int = 100,
    ) -> pd.Series:
        """Connors RSI (ローレンス・コナーズ RSI)"""
        if rsi_periods < 2:
            raise ValueError("rsi_periods must be >= 2")
        if streak_periods < 1:
            raise ValueError("streak_periods must be >= 1")
        if rank_periods < 2:
            raise ValueError("rank_periods must be >= 2")

        max_period = max(rsi_periods, streak_periods, rank_periods)
        validation = validate_series_params(
            close, max_period, min_data_length=max_period
        )
        if validation is not None:
            return pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"CONNORS_RSI_{rsi_periods}_{streak_periods}_{rank_periods}",
            )

        prices = close.astype(float).to_numpy()
        result = OriginalIndicators._connors_rsi_loop(
            prices, rsi_periods, streak_periods, rank_periods
        )

        return pd.Series(
            result,
            index=close.index,
            name=f"CONNORS_RSI_{rsi_periods}_{streak_periods}_{rank_periods}",
        )

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_connors_rsi(data, rsi_periods=3, streak_periods=2, rank_periods=100):
        """Connors RSI計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        connors_rsi = OriginalIndicators.connors_rsi(
            close, rsi_periods, streak_periods, rank_periods
        )

        result = pd.DataFrame(
            {
                connors_rsi.name: connors_rsi,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def gri(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
        offset: int = 0,
    ) -> pd.Series:
        """Gopalakrishnan Range Index (GRI)

        期間内の最高値と最安値のレンジを分析し、市場のボラティリティ/フラクタル特性を測定する。
        GRI = log(max(High, n) - min(Low, n)) / log(n)

        Args:
            high: 高値
            low: 安値
            close: 終値
            length: 期間（デフォルト: 14）
            offset: シフト量

        Returns:
            GRI シリーズ
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            return validation

        # GRI の計算: log(MaxHigh_n - MinLow_n) / log(n)
        hh = high.rolling(window=length).max()
        ll = low.rolling(window=length).min()

        # ゼロ以下にならないよう微小値を加算
        tr = (hh - ll).replace(0, 1e-9)

        result = np.log(tr) / np.log(float(length))

        if offset != 0:
            result = result.shift(offset)

        return result
