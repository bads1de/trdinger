"""Ehlers Cyber Cycle."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_series_params


@njit(cache=True)  # type: ignore[untyped-decorator]
def _njit_cyber_cycle_loop(
    prices: np.ndarray,
    length: int,
    alpha: float,
) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length + 2:
        return result

    # Symmetric triangular smoothing window (1, 2, ..., 2, 1).
    # length=4 gives the classic (1, 2, 2, 1) / 6 coefficients.
    weights = np.empty(length, dtype=np.float64)
    total = 0.0
    for k in range(length):
        w = float(min(k + 1, length - k))
        weights[k] = w
        total += w

    smooth = np.full(n, np.nan, dtype=np.float64)
    for i in range(length - 1, n):
        acc = 0.0
        for k in range(length):
            acc += prices[i - k] * weights[k]
        smooth[i] = acc / total

    cycle = np.full(n, np.nan, dtype=np.float64)

    # Cyber Cycle = (1 - 0.5*alpha)^2 * (smooth - 2*smooth[1] + smooth[2])
    #               + 2*(1-alpha)*cycle[1] - (1-alpha)^2 * cycle[2]
    a2 = 1.0 - alpha
    coeff1 = (1.0 - 0.5 * alpha) ** 2
    coeff2 = 2.0 * a2
    coeff3 = a2**2

    # Seed the recursion with zeros at the first two valid smooth indices.
    start = length - 1
    cycle[start] = 0.0
    cycle[start + 1] = 0.0
    for i in range(start + 2, n):
        cycle[i] = (
            coeff1 * (smooth[i] - 2.0 * smooth[i - 1] + smooth[i - 2])
            + coeff2 * cycle[i - 1]
            - coeff3 * cycle[i - 2]
        )

    for i in range(start, n):
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
    length は平滑化窓(対称三角窓)の幅を制御する。length=4 で
    Ehlers オリジナルの 4点平滑化 (1,2,2,1)/6 と一致する。
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
