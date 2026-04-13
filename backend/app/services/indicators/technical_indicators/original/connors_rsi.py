"""Connors RSI (ローレンス・コナーズ RSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from ...data_validation import handle_pandas_ta_errors, validate_series_params


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
            # 上昇連続: 前日が上昇中なら+1、そうでなければ1から開始
            streaks[i] = streaks[i - 1] + 1.0 if streaks[i - 1] >= 0 else 1.0
        elif prices[i] < prices[i - 1]:
            # 下降連続: 前日が下降中なら-1、そうでなければ-1から開始
            streaks[i] = streaks[i - 1] - 1.0 if streaks[i - 1] <= 0 else -1.0
        else:
            # 同じ価格: リセット
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
