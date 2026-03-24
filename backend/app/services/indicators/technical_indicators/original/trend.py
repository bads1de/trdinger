"""トレンド系の独自テクニカル指標."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
)


@handle_pandas_ta_errors
def chande_kroll_stop(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    p: int = 10,
    x: float = 1.0,
    q: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """Chande Kroll Stop."""
    if p < 1:
        raise ValueError("p must be >= 1")
    if q < 1:
        raise ValueError("q must be >= 1")

    validation = validate_multi_series_params(
        {"high": high, "low": low, "close": close},
        max(p, q),
        min_data_length=max(p, q),
    )
    if validation is not None:
        nan_long = pd.Series(
            np.full(len(close), np.nan), index=close.index, name=f"CKS_LONG_{p}"
        )
        nan_short = pd.Series(
            np.full(len(close), np.nan), index=close.index, name=f"CKS_SHORT_{p}"
        )
        return nan_long, nan_short

    if x <= 0:
        raise ValueError("x must be > 0")

    tr = pd.DataFrame(
        {
            "hl": high - low,
            "hc": abs(high - close.shift(1)),
            "lc": abs(low - close.shift(1)),
        }
    ).max(axis=1)
    atr = tr.rolling(window=p).mean()

    highest_high = high.rolling(window=p).max()
    lowest_low = low.rolling(window=p).min()

    long_stop_initial = highest_high - x * atr
    short_stop_initial = lowest_low + x * atr

    long_stop = long_stop_initial.rolling(window=q).mean()
    short_stop = short_stop_initial.rolling(window=q).mean()

    long_stop.name = f"CKS_LONG_{p}"
    short_stop.name = f"CKS_SHORT_{p}"

    return long_stop, short_stop


@handle_pandas_ta_errors
def calculate_chande_kroll_stop(data, p=10, x=1.0, q=9):
    """Chande Kroll Stop計算のラッパーメソッド."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")

    required_columns = ["high", "low", "close"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    high = data["high"]
    low = data["low"]
    close = data["close"]

    long_stop, short_stop = chande_kroll_stop(high, low, close, p, x, q)

    result = pd.DataFrame(
        {
            long_stop.name: long_stop,
            short_stop.name: short_stop,
        },
        index=data.index,
    )

    return result


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
