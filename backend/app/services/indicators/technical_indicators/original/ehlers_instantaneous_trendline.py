"""Ehlers Instantaneous Trendline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_series_params


@njit(cache=True)
def _njit_instantaneous_trendline_loop(
    prices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < 7:
        return result

    it = np.full(n, np.nan, dtype=np.float64)

    # Initialize: IT needs two seed values
    # it[2] = (price[0] + price[1] + price[2]) / 3
    it[2] = (prices[0] + prices[1] + prices[2]) / 3.0
    # it[3] = (price[0] + 2*price[1] + 2*price[2] + price[3]) / 6
    it[3] = (prices[3] + 2.0 * prices[2] + 2.0 * prices[1] + prices[0]) / 6.0

    for i in range(4, n):
        # IT[i] = (alpha - alpha^2/4)*price[i] + (alpha^2/2)*price[i-1]
        #         - (alpha - 3*alpha^2/4)*price[i-2]
        #         + 2*(1-alpha)*IT[i-1] - (1-alpha)^2*IT[i-2]
        a_sq = alpha * alpha
        it[i] = (
            (alpha - a_sq / 4.0) * prices[i]
            + (a_sq / 2.0) * prices[i - 1]
            - (alpha - 3.0 * a_sq / 4.0) * prices[i - 2]
            + 2.0 * (1.0 - alpha) * it[i - 1]
            - (1.0 - alpha) * (1.0 - alpha) * it[i - 2]
        )

    for i in range(4, n):
        result[i] = it[i]

    return result


@handle_pandas_ta_errors
def ehlers_instantaneous_trendline(
    close: pd.Series,
    alpha: float = 0.07,
) -> pd.Series:
    """Ehlers Instantaneous Trendline.

    John Ehlersのインスタントニアストレンドライン。
    EMAやSMAよりも遅延の小さいトレンドラインを生成する。
    alphaが小さいほど平滑化が強く、遅延が増える。
    """
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be between 0 and 1 exclusive")

    validation = validate_series_params(
        close,
        4,
        min_data_length=7,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"IT_{alpha}",
        )

    result = _njit_instantaneous_trendline_loop(
        close.to_numpy(dtype=float),
        alpha,
    )
    return pd.Series(result, index=close.index, name=f"IT_{alpha}")
