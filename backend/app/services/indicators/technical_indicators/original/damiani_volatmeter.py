"""Damiani Volatmeter."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
)
from ._window_helpers import _window_mean, _window_mean_and_std


@njit(cache=True)
def _njit_damiani_volatmeter_loop(
    high, low, close, vis_atr, vis_std, sed_atr, sed_std
):
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
        s_atr = _window_mean(tr, i - vis_atr + 1, i + 1)
        l_atr = _window_mean(tr, i - sed_atr + 1, i + 1)

        _, std_s = _window_mean_and_std(close, i - vis_std + 1, i + 1)
        _, std_l = _window_mean_and_std(close, i - sed_std + 1, i + 1)

        if std_s > 1e-12 and std_l > 1e-12:
            result[i] = s_atr / std_s - l_atr / std_l

    return result


@handle_pandas_ta_errors
def damiani_volatmeter(
    high,
    low,
    close,
    vis_atr=13,
    vis_std=20,
    sed_atr=40,
    sed_std=100,
    threshold=1.4,
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
        {"high": high, "low": low, "close": close},
        min_data,
        min_data_length=min_data,
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
    osc = pd.Series(
        result, index=close.index, name=f"DAMIANI_{vis_atr}_{sed_atr}"
    )
    thr = pd.Series(
        np.full(len(close), threshold),
        index=close.index,
        name=f"DAMIANI_THR_{threshold}",
    )
    return osc, thr
