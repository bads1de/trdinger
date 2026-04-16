"""Random Walk Index (RWI)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ...data_validation import handle_pandas_ta_errors, validate_multi_series_params


def _calculate_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """True Range を計算する。"""
    return pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)


@handle_pandas_ta_errors
def rwi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> Tuple[pd.Series, pd.Series]:
    """Random Walk Index (RWI)."""
    length = int(length)
    if length < 2:
        raise ValueError("length must be >= 2")

    validation = validate_multi_series_params(
        {"high": high, "low": low, "close": close},
        length,
        min_data_length=length + 1,
    )
    if validation is not None:
        nan_high = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name="RWI_HIGH",
        )
        nan_low = pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name="RWI_LOW",
        )
        return nan_high, nan_low

    high_f = pd.Series(high.values.astype(float), index=high.index, name=high.name)
    low_f = pd.Series(low.values.astype(float), index=low.index, name=low.name)
    close_f = pd.Series(close.values.astype(float), index=close.index, name=close.name)

    true_range = _calculate_true_range(high_f, low_f, close_f)

    high_candidates: list[pd.Series] = []
    low_candidates: list[pd.Series] = []

    for period in range(1, length + 1):
        atr_mean = pd.Series(
            true_range.rolling(window=period, min_periods=period).mean(),
            index=true_range.index,
        )
        atr = atr_mean.shift(1)
        scale = np.sqrt(float(period))

        with np.errstate(divide="ignore", invalid="ignore"):
            high_candidate = (high_f - low_f.shift(period)) / (atr * scale)
            low_candidate = (high_f.shift(period) - low_f) / (atr * scale)

        high_candidates.append(high_candidate.replace([np.inf, -np.inf], np.nan))
        low_candidates.append(low_candidate.replace([np.inf, -np.inf], np.nan))

    rwi_high = pd.concat(high_candidates, axis=1).max(axis=1)
    rwi_low = pd.concat(low_candidates, axis=1).max(axis=1)
    rwi_high.name = "RWI_HIGH"
    rwi_low.name = "RWI_LOW"

    return rwi_high, rwi_low
