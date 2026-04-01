"""オシレーター系の独自テクニカル指標."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit, prange

from ...data_validation import (
    create_nan_series_bundle,
    handle_pandas_ta_errors,
    validate_multi_series_params,
    validate_series_params,
)
from .trend import _format_param


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


def _generate_fibonacci_sequence(count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [1]
    if count == 2:
        return [1, 1]

    sequence = [1, 1]
    for i in range(2, count):
        sequence.append(sequence[i - 1] + sequence[i - 2])
    return sequence


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

    for i in range(max_period + 1, n):
        if np.isfinite(result[i]) and np.isfinite(result[i - 1]):
            result[i] = 0.3 * result[i] + 0.7 * result[i - 1]

    return result


@njit(parallel=True, cache=True)
def _njit_connors_rsi_loop(
    prices: np.ndarray, rsi_periods: int, streak_periods: int, rank_periods: int
) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    max_p = max(rsi_periods, max(streak_periods, rank_periods))

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

    rank_values = np.full(n, np.nan, dtype=np.float64)
    for i in prange(rank_periods, n):
        current_price = prices[i]
        count_lower = 0
        for j in range(i - rank_periods + 1, i + 1):
            if prices[j] <= current_price:
                count_lower += 1
        rank_values[i] = (count_lower / rank_periods) * 100.0

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
    if not primes:
        return create_nan_series_bundle(close, 2)

    primes_array = np.array(primes, dtype=np.int64)
    result = _njit_prime_oscillator_loop(prices, primes_array, 200)

    oscillator = pd.Series(result, index=close.index, name=f"PRIME_OSC_{length}")
    signal = oscillator.rolling(window=signal_length).mean()
    signal.name = f"PRIME_SIGNAL_{length}_{signal_length}"

    return oscillator, signal


@handle_pandas_ta_errors
def calculate_prime_oscillator(data, length=14, signal_length=3):
    """Prime Number Oscillator計算のラッパーメソッド."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")

    required_columns = ["close"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    close = data["close"]
    oscillator, signal = prime_oscillator(close, length, signal_length)

    result = pd.DataFrame(
        {
            oscillator.name: oscillator,
            signal.name: signal,
        },
        index=data.index,
    )

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


@handle_pandas_ta_errors
def calculate_fibonacci_cycle(
    data, cycle_periods: list[int] | None = None, fib_ratios: list[float] | None = None
):
    """Fibonacci Cycle Indicator計算のラッパーメソッド."""
    if isinstance(data, pd.Series):
        close = data
    elif isinstance(data, pd.DataFrame):
        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                if col.capitalize() in data.columns:
                    col = col.capitalize()
                else:
                    raise ValueError(f"Missing required column: {col}")
        close = data[col]
    else:
        raise TypeError("data must be pandas Series or DataFrame")

    cycle, signal = fibonacci_cycle(close, cycle_periods, fib_ratios)

    result = pd.DataFrame(
        {
            cycle.name: cycle,
            signal.name: signal,
        },
        index=close.index,
    )

    return result


@handle_pandas_ta_errors
def adaptive_entropy(
    close: pd.Series,
    short_length: int = 14,
    long_length: int = 28,
    signal_length: int = 5,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Adaptive Entropy Oscillator."""
    if short_length < 5:
        raise ValueError("short_length must be >= 5")
    if long_length < 10:
        raise ValueError("long_length must be >= 10")
    if signal_length < 2:
        raise ValueError("signal_length must be >= 2")
    if short_length >= long_length:
        raise ValueError("short_length must be < long_length")

    validation = validate_series_params(close, long_length, min_data_length=long_length)
    if validation is not None:
        nan_osc = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"ADAPTIVE_ENTROPY_OSC_{short_length}_{long_length}",
        )
        nan_sig = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=(
                f"ADAPTIVE_ENTROPY_SIGNAL_{short_length}_{long_length}_{signal_length}"
            ),
        )
        nan_ratio = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"ADAPTIVE_ENTROPY_RATIO_{short_length}_{long_length}",
        )
        return nan_osc, nan_sig, nan_ratio

    prices = close.astype(float).to_numpy()
    short_entropy = _njit_entropy_loop(prices, short_length)
    long_entropy = _njit_entropy_loop(prices, long_length)

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_ratio = short_entropy / long_entropy

    normalized_osc = (entropy_ratio - 0.5) * 2.0

    signal = (
        pd.Series(normalized_osc, index=close.index)
        .rolling(window=signal_length)
        .mean()
    )

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


@handle_pandas_ta_errors
def calculate_adaptive_entropy(data, short_length=14, long_length=28, signal_length=5):
    """Adaptive Entropy Oscillator計算のラッパーメソッド."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")

    required_columns = ["close"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    close = data["close"]
    oscillator, signal, ratio = adaptive_entropy(
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


@handle_pandas_ta_errors
def connors_rsi(
    close: pd.Series,
    rsi_periods: int = 3,
    streak_periods: int = 2,
    rank_periods: int = 100,
) -> pd.Series:
    """Connors RSI (ローレンス・コナーズ RSI)."""
    if rsi_periods < 2:
        raise ValueError("rsi_periods must be >= 2")
    if streak_periods < 1:
        raise ValueError("streak_periods must be >= 1")
    if rank_periods < 2:
        raise ValueError("rank_periods must be >= 2")

    max_period = max(rsi_periods, streak_periods, rank_periods)
    validation = validate_series_params(close, max_period, min_data_length=max_period)
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"CONNORS_RSI_{rsi_periods}_{streak_periods}_{rank_periods}",
        )

    prices = close.astype(float).to_numpy()
    result = _njit_connors_rsi_loop(prices, rsi_periods, streak_periods, rank_periods)

    return pd.Series(
        result,
        index=close.index,
        name=f"CONNORS_RSI_{rsi_periods}_{streak_periods}_{rank_periods}",
    )


@handle_pandas_ta_errors
def calculate_connors_rsi(data, rsi_periods=3, streak_periods=2, rank_periods=100):
    """Connors RSI計算のラッパーメソッド."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")

    required_columns = ["close"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    close = data["close"]
    connors_rsi_result = connors_rsi(close, rsi_periods, streak_periods, rank_periods)

    result = pd.DataFrame(
        {
            connors_rsi_result.name: connors_rsi_result,
        },
        index=data.index,
    )

    return result


@njit(cache=True)
def _njit_damiani_volatmeter_loop(high, low, close, vis_atr, vis_std, sed_atr, sed_std):
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    min_len = max(sed_atr, sed_std)
    if n < min_len + 1:
        return result

    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            max(abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])),
        )

    for i in range(min_len, n):
        s_atr = 0.0
        for j in range(i - vis_atr + 1, i + 1):
            s_atr += tr[j]
        s_atr /= float(vis_atr)

        l_atr = 0.0
        for j in range(i - sed_atr + 1, i + 1):
            l_atr += tr[j]
        l_atr /= float(sed_atr)

        sma_s = 0.0
        for j in range(i - vis_std + 1, i + 1):
            sma_s += close[j]
        sma_s /= float(vis_std)
        var_s = 0.0
        for j in range(i - vis_std + 1, i + 1):
            d = close[j] - sma_s
            var_s += d * d
        std_s = np.sqrt(var_s / float(vis_std))

        sma_l = 0.0
        for j in range(i - sed_std + 1, i + 1):
            sma_l += close[j]
        sma_l /= float(sed_std)
        var_l = 0.0
        for j in range(i - sed_std + 1, i + 1):
            d = close[j] - sma_l
            var_l += d * d
        std_l = np.sqrt(var_l / float(sed_std))

        if std_s > 1e-12 and std_l > 1e-12:
            result[i] = s_atr / std_s - l_atr / std_l

    return result


@handle_pandas_ta_errors
def damiani_volatmeter(
    high, low, close, vis_atr=13, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4
):
    """Damiani Volatmeter.

    Compares short-term to long-term volatility to filter market conditions.
    Low values = noisy/choppy (avoid trading), High values = sufficient volatility.

    Args:
        high: High price series. low: Low price series. close: Close price series.
        vis_atr/vis_std: Short-term periods. sed_atr/sed_std: Long-term periods.
        threshold: Volatility threshold. Default 1.4.
    Returns:
        Tuple of (volatmeter, threshold_line).
    """
    min_data = max(sed_atr, sed_std) + 1
    validation = validate_multi_series_params(
        {"high": high, "low": low, "close": close}, min_data, min_data_length=min_data
    )
    if validation is not None:
        nan1 = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"DAMIANI_{vis_atr}_{sed_atr}",
        )
        nan2 = pd.Series(
            np.full(len(close), threshold),
            index=close.index,
            name=f"DAMIANI_THR_{threshold}",
        )
        return nan1, nan2

    result = _njit_damiani_volatmeter_loop(
        high.values.astype(float),
        low.values.astype(float),
        close.values.astype(float),
        vis_atr,
        vis_std,
        sed_atr,
        sed_std,
    )
    osc = pd.Series(result, index=close.index, name=f"DAMIANI_{vis_atr}_{sed_atr}")
    thr = pd.Series(
        np.full(len(close), threshold),
        index=close.index,
        name=f"DAMIANI_THR_{threshold}",
    )
    return osc, thr


@handle_pandas_ta_errors
def calculate_damiani_volatmeter(
    data, vis_atr=13, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4
):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")
    for col in ["high", "low", "close"]:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    osc, thr = damiani_volatmeter(
        data["high"],
        data["low"],
        data["close"],
        vis_atr,
        vis_std,
        sed_atr,
        sed_std,
        threshold,
    )
    return pd.DataFrame({osc.name: osc, thr.name: thr}, index=data.index)


@njit(cache=True)
def _njit_kairi_loop(prices, length):
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return result
    for i in range(length - 1, n):
        sma = 0.0
        for j in range(i - length + 1, i + 1):
            sma += prices[j]
        sma /= float(length)
        if abs(sma) > 1e-12:
            result[i] = ((prices[i] - sma) / sma) * 100.0
    return result


@handle_pandas_ta_errors
def kairi_relative_index(close, length=14, signal_length=3):
    """Kairi Relative Index (KRI).

    Percentage deviation of price from its moving average.
    Positive = above MA, negative = below MA.

    Args:
        close: Close price series. length: SMA period. Default 14.
    Returns:
        Tuple of (kri, signal).
    """
    validation = validate_series_params(close, length, min_data_length=length)
    if validation is not None:
        nan1 = pd.Series(
            np.full(len(close), np.nan), index=close.index, name=f"KRI_{length}"
        )
        nan2 = pd.Series(
            np.full(len(close), np.nan), index=close.index, name=f"KRI_SIGNAL_{length}"
        )
        return nan1, nan2

    result = _njit_kairi_loop(close.values.astype(float), length)
    osc = pd.Series(result, index=close.index, name=f"KRI_{length}")
    sig = osc.rolling(window=signal_length, min_periods=1).mean()
    sig.name = f"KRI_SIGNAL_{length}"
    return osc, sig


@handle_pandas_ta_errors
def calculate_kairi_relative_index(data, length=14, signal_length=3):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")
    if "close" not in data.columns:
        raise ValueError("Missing required column: close")
    kri, sig = kairi_relative_index(data["close"], length, signal_length)
    return pd.DataFrame({kri.name: kri, sig.name: sig}, index=data.index)


@njit(parallel=True, cache=True)
def _njit_entropy_volatility_loop(
    returns: np.ndarray,
    win: int,
    m_val: int,
    r_val: float,
) -> np.ndarray:
    n = len(returns)
    res = np.full(n, np.nan)
    for i in prange(win - 1, n):
        chunk = returns[i - win + 1 : i + 1]
        # 標準偏差を計算
        mean = 0.0
        for val in chunk:
            mean += val
        mean /= len(chunk)
        variance = 0.0
        for val in chunk:
            variance += (val - mean) ** 2
        std = np.sqrt(variance / len(chunk))
        if std < 1e-12:
            res[i] = 0.0
            continue
        thresh = r_val * std
        # サンプルエントロピー計算
        a_count = 0
        b_count = 0
        for j in range(len(chunk) - m_val):
            for k in range(j + 1, len(chunk) - m_val):
                # m_val次元の距離を計算
                match_m = True
                for m in range(m_val):
                    if abs(chunk[j + m] - chunk[k + m]) > thresh:
                        match_m = False
                        break
                if match_m:
                    b_count += 1
                    # m_val+1次元の距離を計算
                    if abs(chunk[j + m_val] - chunk[k + m_val]) <= thresh:
                        a_count += 1
        if a_count > 0 and b_count > 0:
            res[i] = -np.log(a_count / b_count)
        else:
            res[i] = 0.0
    return res


@handle_pandas_ta_errors
def entropy_volatility_index(
    close: pd.Series,
    length: int = 30,
    m_val: int = 2,
    r_val: float = 0.2,
) -> pd.Series:
    """エントロピーボラティリティインデックス (EVI)."""
    if length < 1:
        raise ValueError("length must be >= 1")
    if m_val < 1:
        raise ValueError("m_val must be >= 1")
    if r_val <= 0:
        raise ValueError("r_val must be > 0")

    validation = validate_series_params(close, length)
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"EVI_{length}_{m_val}_{_format_param(r_val)}",
        )

    returns = np.log(close / close.shift(1)).to_numpy()
    returns[0] = 0.0
    evi = _njit_entropy_volatility_loop(returns, length, m_val, r_val)
    return pd.Series(
        evi,
        index=close.index,
        name=f"EVI_{length}_{m_val}_{_format_param(r_val)}",
    )


@handle_pandas_ta_errors
def calculate_entropy_volatility_index(data, length=30, m_val=2, r_val=0.2):
    """エントロピーボラティリティインデックス計算のラッパーメソッド."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")

    required_columns = ["close"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    close = data["close"]
    evi = entropy_volatility_index(
        close, length=length, m_val=m_val, r_val=r_val
    )

    result = pd.DataFrame({evi.name: evi}, index=data.index)
    return result
