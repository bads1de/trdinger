"""Gopalakrishnan Range Index (GRI)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
)


@handle_pandas_ta_errors
def gri(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    offset: int = 0,
) -> pd.Series:
    """Gopalakrishnan Range Index (GRI)."""
    validation = validate_multi_series_params(
        {"high": high, "low": low, "close": close}, length
    )
    if validation is not None:
        return validation

    hh = high.rolling(window=length).max()
    ll = low.rolling(window=length).min()
    tr = (hh - ll).replace(0, 1e-9)
    result = np.log(tr) / np.log(float(length))

    if offset != 0:
        result = result.shift(offset)

    return result