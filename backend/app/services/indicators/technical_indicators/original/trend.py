"""トレンド系の独自テクニカル指標."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from ...data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
    validate_series_params,
)


def _format_param(value: float | int) -> str:
    """名前用に数値を短く整形する。"""
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:g}"


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


@handle_pandas_ta_errors
def calculate_trend_intensity_index(data, length=14, sma_length=30):
    """Trend Intensity Index計算のラッパーメソッド."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")

    required_columns = ["close", "high", "low"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    close = data["close"]
    high = data["high"]
    low = data["low"]

    tii = trend_intensity_index(close, high, low, length, sma_length)

    result = pd.DataFrame({tii.name: tii}, index=data.index)
    return result


@njit(parallel=True, cache=True)
def _njit_direction_entropy_loop(
    directions: np.ndarray,
    win: int,
    m_val: int,
) -> np.ndarray:
    n = len(directions)
    res = np.full(n, np.nan)
    for i in prange(win - 1, n):
        chunk = directions[i - win + 1 : i + 1]
        a_count = 0
        b_count = 0
        for j in range(len(chunk) - m_val):
            for k in range(j + 1, len(chunk) - m_val):
                match_m = True
                for m in range(m_val):
                    if chunk[j + m] != chunk[k + m]:
                        match_m = False
                        break
                if match_m:
                    b_count += 1
                    if chunk[j + m_val] == chunk[k + m_val]:
                        a_count += 1
        if a_count > 0 and b_count > 0:
            res[i] = -np.log(a_count / b_count)
        else:
            res[i] = 0.0
    return res


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


@handle_pandas_ta_errors
def direction_entropy(
    close: pd.Series,
    length: int = 14,
    m_val: int = 2,
) -> pd.Series:
    """方向エントロピーインデックス (DEI)."""
    if length < 1:
        raise ValueError("length must be >= 1")
    if m_val < 1:
        raise ValueError("m_val must be >= 1")

    validation = validate_series_params(close, length)
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"DEI_{length}_{m_val}",
        )

    diff = close.diff()
    directions = np.where(diff > 0, 1, np.where(diff < 0, -1, 0)).astype(np.float64)
    dei = _njit_direction_entropy_loop(directions, length, m_val)
    return pd.Series(dei, index=close.index, name=f"DEI_{length}_{m_val}")


@handle_pandas_ta_errors
def calculate_direction_entropy(data, length=14, m_val=2):
    """方向エントロピーインデックス計算のラッパーメソッド."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")

    required_columns = ["close"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    close = data["close"]
    dei = direction_entropy(close, length=length, m_val=m_val)

    result = pd.DataFrame({dei.name: dei}, index=data.index)
    return result
