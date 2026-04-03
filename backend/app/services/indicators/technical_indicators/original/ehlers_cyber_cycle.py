"""Ehlers Cyber Cycle."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_series_params


@njit(cache=True)
def _njit_cyber_cycle_loop(
    prices: np.ndarray,
    length: int,
    alpha: float,
) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < 7:
        return result

    smooth = np.full(n, np.nan, dtype=np.float64)
    cycle = np.full(n, np.nan, dtype=np.float64)

    # Smooth = (price + 2*price[1] + 2*price[2] + price[3]) / 6
    for i in range(3, n):
        smooth[i] = (
            prices[i] + 2.0 * prices[i - 1] + 2.0 * prices[i - 2] + prices[i - 3]
        ) / 6.0

    # Cyber Cycle = (1 - 0.5*alpha)^2 * (smooth - 2*smooth[1] + smooth[2])
    #               + 2*(1-alpha)*cycle[1] - (1-alpha)^2 * cycle[2]
    a2 = 1.0 - alpha
    coeff1 = (1.0 - 0.5 * alpha) ** 2
    coeff2 = 2.0 * a2
    coeff3 = a2**2

    # Initialize first valid cycle value at index 5
    # (needs smooth[3], smooth[4], smooth[5] which are all valid)
    # Use simple difference as seed for cycle[4]
    cycle[4] = (smooth[4] - smooth[3]) * 0.5
    # cycle[5] from the recursive formula using cycle[4] and a pseudo cycle[3]
    cycle[5] = (
        coeff1 * (smooth[5] - 2.0 * smooth[4] + smooth[3])
        + coeff2 * cycle[4]
        - coeff3 * 0.0  # assume cycle[3] = 0 for seed
    )

    for i in range(6, n):
        cycle[i] = (
            coeff1 * (smooth[i] - 2.0 * smooth[i - 1] + smooth[i - 2])
            + coeff2 * cycle[i - 1]
            - coeff3 * cycle[i - 2]
        )

    # Fill result from index 4 onward
    for i in range(4, n):
        result[i] = cycle[i]

    return result


@handle_pandas_ta_errors
def ehlers_cyber_cycle(
    close: pd.Series,
    length: int = 14,
    alpha: float = 0.07,
) -> pd.Series:
    """Ehlers Cyber Cycle.

    John Ehlersのサイバー・サイクルアルゴリズム。
    市場サイクルを検出し、トレンドとサイクルの分離を行う。
    alphaが小さいほど平滑化が強く、遅延が増える。
    """
    if length < 2:
        raise ValueError("length must be >= 2")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be between 0 and 1 exclusive")

    validation = validate_series_params(
        close,
        length,
        min_data_length=length + 4,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"CC_{length}",
        )

    result = _njit_cyber_cycle_loop(
        close.to_numpy(dtype=float),
        length,
        alpha,
    )
    return pd.Series(result, index=close.index, name=f"CC_{length}")
