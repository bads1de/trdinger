"""Vortex RSI Hybrid (VRSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ...data_validation import handle_pandas_ta_errors, validate_multi_series_params


@njit(cache=True)
def _njit_vortex_rsi_loop(
    high_values: np.ndarray,
    low_values: np.ndarray,
    close_values: np.ndarray,
    length: int,
) -> np.ndarray:
    n = len(high_values)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length + 1:
        return result

    # True Range
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        hl = high_values[i] - low_values[i]
        hc = abs(high_values[i] - close_values[i - 1])
        lc = abs(low_values[i] - close_values[i - 1])
        tr[i] = max(hl, max(hc, lc))

    # Vortex Movement
    vm_plus = np.zeros(n, dtype=np.float64)
    vm_minus = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        vm_plus[i] = abs(high_values[i] - low_values[i - 1])
        vm_minus[i] = abs(low_values[i] - high_values[i - 1])

    # Sum over period
    for i in range(length, n):
        sum_tr = 0.0
        sum_vm_plus = 0.0
        sum_vm_minus = 0.0
        for j in range(i - length + 1, i + 1):
            sum_tr += tr[j]
            sum_vm_plus += vm_plus[j]
            sum_vm_minus += vm_minus[j]

        if sum_tr > 1e-12:
            vi_plus = sum_vm_plus / sum_tr
            vi_minus = sum_vm_minus / sum_tr

            # RSI-like transformation: VM+ - VM- normalized to 0-100
            diff = vi_plus - vi_minus
            sum_vi = vi_plus + vi_minus
            if sum_vi > 1e-12:
                result[i] = (diff / sum_vi + 1.0) * 50.0
            else:
                result[i] = 50.0
        else:
            result[i] = 50.0

    return result


@handle_pandas_ta_errors
def vortex_rsi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Vortex RSI Hybrid (VRSI).

    Vortex IndicatorとRSIのハイブリッド指標。
    ボルテックスムーブメントをRSIスタイルの0-100オシレーターに変換する。
    上昇トレンドの強さ(50以上)と下降トレンドの強さ(50以下)を示す。
    """
    if length < 1:
        raise ValueError("length must be >= 1")

    validation = validate_multi_series_params(
        {"high": high, "low": low, "close": close},
        length,
        min_data_length=length + 1,
    )
    if validation is not None:
        return pd.Series(
            np.full(len(close), np.nan),
            index=close.index,
            name=f"VRSI_{length}",
        )

    result = _njit_vortex_rsi_loop(
        high.to_numpy(dtype=float),
        low.to_numpy(dtype=float),
        close.to_numpy(dtype=float),
        length,
    )
    return pd.Series(result, index=close.index, name=f"VRSI_{length}")
