"""Market Meanness Index (MMI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_series_params


@njit(cache=True)
def _njit_mmi_loop(prices: np.ndarray, length: int) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length or length < 3:
        return result

    denominator = float(length - 2)
    for i in range(length - 1, n):
        start = i - length + 1
        count = 0

        for j in range(start + 2, i + 1):
            prev_change = prices[j - 1] - prices[j - 2]
            curr_change = prices[j] - prices[j - 1]
            if prev_change * curr_change < 0.0:
                count += 1

        result[i] = (count / denominator) * 100.0

    return result


@handle_pandas_ta_errors
def mmi(close: pd.Series, length: int = 20) -> pd.Series:
    """Market Meanness Index (MMI)."""
    if length < 3:
        raise ValueError("length must be >= 3")

    validation = validate_series_params(
        close,
        length,
        min_data_length=length,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"MMI_{length}",
        )

    result = _njit_mmi_loop(close.astype(float).to_numpy(), length)
    return pd.Series(result, index=close.index, name=f"MMI_{length}")
