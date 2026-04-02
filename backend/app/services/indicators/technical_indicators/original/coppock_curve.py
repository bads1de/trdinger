"""Coppock Curve."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_series_params


@njit(cache=True)
def _njit_coppock_loop(
    prices: np.ndarray,
    long_roc: int,
    short_roc: int,
    wma_length: int,
) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    max_roc = long_roc if long_roc > short_roc else short_roc
    min_len = max_roc + wma_length
    if n < min_len:
        return result

    # Compute ROC_sum = ROC_short + ROC_long
    roc_sum = np.full(n, np.nan, dtype=np.float64)
    for i in range(max_roc, n):
        roc_s = 0.0
        if i >= short_roc and prices[i - short_roc] > 1e-12:
            roc_s = (prices[i] - prices[i - short_roc]) / prices[i - short_roc] * 100.0
        roc_l = 0.0
        if i >= long_roc and prices[i - long_roc] > 1e-12:
            roc_l = (prices[i] - prices[i - long_roc]) / prices[i - long_roc] * 100.0
        roc_sum[i] = roc_s + roc_l

    # WMA of roc_sum
    for i in range(max_roc + wma_length - 1, n):
        total = 0.0
        weight_sum = 0.0
        for j in range(wma_length):
            weight = float(j + 1)
            total += roc_sum[i - wma_length + 1 + j] * weight
            weight_sum += weight
        if weight_sum > 1e-12:
            result[i] = total / weight_sum

    return result


@handle_pandas_ta_errors
def coppock_curve(
    close: pd.Series,
    long_roc: int = 14,
    short_roc: int = 11,
    wma_length: int = 10,
) -> pd.Series:
    """Coppock Curve (Coppock Indicator).

    長期投資の買いシグナルを検出するモメンタム指標。
    ROC(Long) + ROC(Short)のWMAで計算される。
    ゼロライン以下から上方へのクロスオーバーが買いシグナル。
    """
    if long_roc < 1:
        raise ValueError("long_roc must be >= 1")
    if short_roc < 1:
        raise ValueError("short_roc must be >= 1")
    if wma_length < 1:
        raise ValueError("wma_length must be >= 1")

    max_roc = max(long_roc, short_roc)
    validation = validate_series_params(
        close,
        wma_length,
        min_data_length=max_roc + wma_length,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"COPPOCK_{long_roc}_{short_roc}_{wma_length}",
        )

    result = _njit_coppock_loop(
        close.to_numpy(dtype=float),
        long_roc,
        short_roc,
        wma_length,
    )
    return pd.Series(
        result,
        index=close.index,
        name=f"COPPOCK_{long_roc}_{short_roc}_{wma_length}",
    )
