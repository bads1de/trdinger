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
    # 原著(Gopalakrishnan)定義: ln(HH/LL) / ln(N)
    # 比方式は価格スケールに依存せず、異なる価格帯の銘柄間で比較可能。
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(hh / ll.replace(0, np.nan)) / np.log(float(length))
    result = result.replace([np.inf, -np.inf], np.nan)

    if offset != 0:
        result = result.shift(offset)

    return result
