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


def _clamp_confidence(confidence: float) -> float:
    """信頼区間を[0, 0.999]の範囲に制限"""
    return float(min(max(confidence, 0.0), 0.999))


def calculate_historical_var(returns: Iterable[float], confidence: float) -> float:
    """ヒストリカルVaR（損失率）を計算"""
    prepared = _prepare_returns(returns)
    if prepared.size == 0: return 0.0

    quantile = np.quantile(prepared, 1 - _clamp_confidence(confidence))
    return float(abs(min(quantile, 0.0)))


def calculate_expected_shortfall(returns: Iterable[float], confidence: float) -> float:
    """ヒストリカルES（条件付平均損失率）を計算"""
    prepared = _prepare_returns(returns)
    if prepared.size == 0: return 0.0

    threshold = np.quantile(prepared, 1 - _clamp_confidence(confidence))
    tail_losses = prepared[prepared <= threshold]
    if tail_losses.size == 0: return 0.0

    return float(abs(np.minimum(tail_losses, 0.0).mean()))





