"""エントロピーボラティリティインデックス (EVI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from ...data_validation import handle_pandas_ta_errors, validate_series_params
from ._window_helpers import _window_mean_and_std


@njit(parallel=True, cache=True)
def _njit_entropy_volatility_loop(
    returns: np.ndarray,
    win: int,
    m_val: int,
    r_val: float,
) -> np.ndarray:
    n = len(returns)
    res = np.full(n, np.nan)
    for i in prange(win - 1, n):
        chunk = returns[i - win + 1 : i + 1]
        _, std = _window_mean_and_std(chunk, 0, len(chunk))
        if std < 1e-12:
            res[i] = 0.0
            continue
        thresh = r_val * std
        # サンプルエントロピー計算
        a_count = 0
        b_count = 0
        for j in range(len(chunk) - m_val):
            for k in range(j + 1, len(chunk) - m_val):
                # m_val次元の距離を計算
                match_m = True
                for m in range(m_val):
                    if abs(chunk[j + m] - chunk[k + m]) > thresh:
                        match_m = False
                        break
                if match_m:
                    b_count += 1
                    # m_val+1次元の距離を計算
                    if abs(chunk[j + m_val] - chunk[k + m_val]) <= thresh:
                        a_count += 1
        if a_count > 0 and b_count > 0:
            res[i] = -np.log(a_count / b_count)
        else:
            res[i] = 0.0
    return res


@handle_pandas_ta_errors
def entropy_volatility_index(
    close: pd.Series,
    length: int = 30,
    m_val: int = 2,
    r_val: float = 0.2,
) -> pd.Series:
    """エントロピーボラティリティインデックス (EVI)."""
    if length < 1:
        raise ValueError("length must be >= 1")
    if m_val < 1:
        raise ValueError("m_val must be >= 1")
    if r_val <= 0:
        raise ValueError("r_val must be > 0")

    validation = validate_series_params(close, length)
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"EVI_{length}_{m_val}_{r_val:g}",
        )

    from typing import cast

    returns = cast(pd.Series, np.log(close / close.shift(1))).to_numpy(dtype=np.float64, copy=True)
    returns[0] = 0.0
    evi = _njit_entropy_volatility_loop(returns, length, m_val, r_val)
    return pd.Series(
        evi,
        index=close.index,
        name=f"EVI_{length}_{m_val}_{r_val:g}",
    )
