"""Trend Intensity Index (TII)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
)


@handle_pandas_ta_errors
def trend_intensity_index(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    length: int = 14,
    sma_length: int = 30,
) -> pd.Series:
    """Trend Intensity Index (TII)."""
    if length < 1:
        raise ValueError("length must be >= 1")
    if sma_length < 1:
        raise ValueError("sma_length must be >= 1")

    validation = validate_multi_series_params(
        {"close": close, "high": high, "low": low},
        max(length, sma_length),
        min_data_length=max(length, sma_length),
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"TII_{length}_{sma_length}",
        )

    sma = close.rolling(window=sma_length).mean()
    above_sma = (close > sma).astype(int)
    count_above = above_sma.rolling(window=length).sum()
    tii = (count_above / length) * 100

    return pd.Series(tii, index=close.index, name=f"TII_{length}_{sma_length}")