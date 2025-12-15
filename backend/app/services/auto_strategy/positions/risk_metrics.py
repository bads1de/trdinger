"""リスク指標計算ユーティリティ"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _prepare_returns(raw_returns: Iterable[float]) -> np.ndarray:
    """計算用にリターン配列を整形"""

    if raw_returns is None:
        return np.array([], dtype=float)

    try:
        array = np.asarray(list(raw_returns), dtype=float)
    except TypeError:
        array = np.array([], dtype=float)

    if array.size == 0:
        return array

    return array[np.isfinite(array)]


def calculate_historical_var(returns: Iterable[float], confidence: float) -> float:
    """ヒストリカルVaR（損失率）を計算"""

    prepared = _prepare_returns(returns)
    if prepared.size == 0:
        return 0.0

    clamped_confidence = float(min(max(confidence, 0.0), 0.999))
    quantile = np.quantile(prepared, 1 - clamped_confidence)
    return float(abs(min(quantile, 0.0)))


def calculate_expected_shortfall(returns: Iterable[float], confidence: float) -> float:
    """ヒストリカルES（条件付平均損失率）を計算"""

    prepared = _prepare_returns(returns)
    if prepared.size == 0:
        return 0.0

    clamped_confidence = float(min(max(confidence, 0.0), 0.999))
    threshold = np.quantile(prepared, 1 - clamped_confidence)
    tail_losses = prepared[prepared <= threshold]
    if tail_losses.size == 0:
        return 0.0

    clipped = np.minimum(tail_losses, 0.0)
    return float(abs(clipped.mean()))


