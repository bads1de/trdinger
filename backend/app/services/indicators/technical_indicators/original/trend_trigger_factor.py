"""Trend Trigger Factor (TTF)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
)


@handle_pandas_ta_errors
def ttf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 15,
) -> pd.Series:
    """Trend Trigger Factor (TTF)."""
    length = int(length)
    if length < 2:
        raise ValueError("length must be >= 2")

    validation = validate_multi_series_params(
        {"high": high, "low": low, "close": close},
        length,
        min_data_length=2 * length,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"TTF_{length}",
        )

    high_f = high.astype(float)
    low_f = low.astype(float)

    current_high = high_f.rolling(window=length, min_periods=length).max()
    current_low = low_f.rolling(window=length, min_periods=length).min()
    prev_high = current_high.shift(length)
    prev_low = current_low.shift(length)

    buy_power = current_high - prev_low
    sell_power = prev_high - current_low
    denominator = 0.5 * (buy_power + sell_power)

    result = pd.Series(np.nan, index=high.index, name=f"TTF_{length}")
    finite_mask = denominator.notna()
    safe_mask = finite_mask & (denominator.abs() > 1e-12)
    zero_mask = finite_mask & ~safe_mask

    with np.errstate(divide="ignore", invalid="ignore"):
        result.loc[safe_mask] = (
            (buy_power.loc[safe_mask] - sell_power.loc[safe_mask])
            / denominator.loc[safe_mask]
        ) * 100.0

    if bool(zero_mask.any()):
        result.loc[zero_mask] = 0.0

    return result
