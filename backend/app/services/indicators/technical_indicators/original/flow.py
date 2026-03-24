"""フロー・カオス系の独自テクニカル指標."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit, prange

from ...data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
)


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
def _njit_dft_magnitude(x: np.ndarray) -> np.ndarray:
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
    n = len(prices)
    if n < 4:
        return np.zeros(0)

    windowed = np.empty(n, dtype=np.float64)
    for i in range(n):
        w = 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (n - 1))
        windowed[i] = prices[i] * w

    magnitude = _njit_dft_magnitude(windowed)

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

    p_indices = np.array(peaks_indices)
    p_mags = np.array(peaks_mags)
    sorted_idx = np.argsort(p_mags)[::-1]

    num_peaks = min(3, len(sorted_idx))
    res_freqs = np.empty(num_peaks, dtype=np.float64)
    for i in range(num_peaks):
        res_freqs[i] = float(p_indices[sorted_idx[i]]) / float(n)

    return res_freqs


@njit(cache=True)
def _njit_apply_bandpass_res(x: np.ndarray, freq: float, q: float = 2.0) -> np.ndarray:
    n = len(x)
    y = np.zeros(n, dtype=np.float64)

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

    for i in prange(length, n):
        wavelet_component = wavelet_result[i]
        if not np.isfinite(wavelet_component):
            wavelet_component = 0.0

        corr_component = correlation_score[i]
        vol_component = volatility[i]

        integrated = wavelet_component * 0.4 + corr_component * 0.3 + vol_component * 0.3

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


def _chaos_fractal_dimension_loop(
    prices: np.ndarray, volumes: np.ndarray, length: int, embedding_dim: int
) -> np.ndarray:
    min_period = max(length, 30)
    raw_scores = _njit_ctfd_raw_pass(prices, volumes, length, embedding_dim, min_period)
    return _njit_ctfd_normalize(raw_scores, min_period)


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

    signal = pd.Series(flow_values, index=close.index).rolling(window=flow_length).mean()
    signal.name = "QUANTUM_FLOW_SIGNAL"

    flow_series = pd.Series(flow_values, index=close.index, name="QUANTUM_FLOW")
    return flow_series, signal


@handle_pandas_ta_errors
def calculate_quantum_flow(data, length=14, flow_length=9):
    """Quantum Flow Analysis計算のラッパーメソッド."""
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

    flow, signal = quantum_flow(close, high, low, volume, length, flow_length)

    result = pd.DataFrame(
        {
            flow.name: flow,
            signal.name: signal,
        },
        index=data.index,
    )

    return result


@handle_pandas_ta_errors
def harmonic_resonance(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    length: int = 20,
    resonance_bands: int = 5,
    signal_length: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Harmonic Resonance Indicator (HRI)."""
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
    hri_values = _njit_harmonic_resonance_loop(
        prices, length, resonance_bands, min_period
    )

    hri_series = pd.Series(hri_values, index=close.index, name="HARMONIC_RESONANCE")
    signal = hri_series.rolling(window=signal_length, min_periods=1).mean()
    signal.name = "HRI_SIGNAL"

    return hri_series, signal


@handle_pandas_ta_errors
def calculate_harmonic_resonance(
    data, length=20, resonance_bands=5, signal_length=3
):
    """Harmonic Resonance Indicator計算のラッパーメソッド."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")

    required_columns = ["close", "high", "low"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    close = data["close"]
    high = data["high"]
    low = data["low"]

    hri, signal = harmonic_resonance(close, high, low, length, resonance_bands, signal_length)

    result = pd.DataFrame(
        {
            hri.name: hri,
            signal.name: signal,
        },
        index=data.index,
    )

    return result


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
    signal.name = "CTFD_SIGNAL"

    return ctf_series, signal


@handle_pandas_ta_errors
def calculate_chaos_fractal_dimension(
    data, length=25, embedding_dim=3, signal_length=4
):
    """Chaos Theory Fractal Dimension計算のラッパーメソッド."""
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

    ctf, signal = chaos_fractal_dimension(
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
