"""Harmonic Resonance Indicator (HRI)."""

from __future__ import annotations

from typing import Tuple, cast

import numpy as np
import pandas as pd
from numba import njit, prange

from ...data_validation import handle_pandas_ta_errors, validate_multi_series_params
from ._window_helpers import _window_mean


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

        mean_p = _window_mean(price_window, 0, length)

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
                f_mean = _window_mean(filtered, 0, length)

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
    signal.name = "HRI_SIGNAL"  # type: ignore[reportAttributeAccessIssue]

    return cast(Tuple[pd.Series, pd.Series], (hri_series, signal))
