"""
評価指標モジュール

リスク指標（Ulcer Index、取引頻度ペナルティ）の計算を提供します。
"""

import functools
import logging
import math
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from numba import njit

from app.utils.datetime_utils import parse_datetime_optional

logger = logging.getLogger(__name__)

# 基準となる1日あたりの取引回数（これを超えると過剰取引とみなすペナルティが増加）
REFERENCE_TRADES_PER_DAY = 8.0


@functools.lru_cache(maxsize=1024)
def _ensure_datetime(value: Optional[object]) -> Optional[datetime]:
    """値をdatetimeオブジェクトに変換します（キャッシュ付き）。"""
    return parse_datetime_optional(value)


@njit(cache=True)
def _calculate_ulcer_index_numba(dd_array: np.ndarray) -> float:
    """
    NumbaでUlcer Indexの数値計算を高速化
    """
    if len(dd_array) == 0:
        return 0.0

    # 二乗和を計算
    squared_sum = 0.0
    count = 0
    for i in range(len(dd_array)):
        val = dd_array[i]
        if np.isnan(val):
            continue

        # 絶対値化
        val = abs(val)

        # 1.0より大きい場合はパーセンテージ(0-100)とみなして小数(0-1.0)に正規化
        if val > 1.0:
            val /= 100.0

        squared_sum += val * val
        count += 1

    if count == 0:
        return 0.0

    return np.sqrt(squared_sum / count)


def calculate_ulcer_index(equity_curve: Sequence[Mapping[str, Any]]) -> float:
    """
    Ulcer Index（ドローダウンの二乗平均平方根）を計算します。
    標準偏差よりも下落リスクを重視するリスク指標です。

    Args:
        equity_curve: バックテスト結果の資産曲線のポイント列。

    Returns:
        ドローダウン率の二乗平均平方根（小数値）。
    """
    if not equity_curve:
        return 0.0

    try:
        dd_array = np.array(
            [
                float(p.get("drawdown", 0.0) or 0.0) if isinstance(p, Mapping) else 0.0
                for p in equity_curve
            ],
            dtype=np.float64,
        )

        return _calculate_ulcer_index_numba(dd_array)

    except Exception as e:
        logger.warning(f"Ulcer Index計算エラー: {e}")
        return 0.0


def calculate_trade_frequency_penalty(
    *,
    total_trades: int,
    start_date: Optional[object],
    end_date: Optional[object],
    trade_history: Optional[Iterable[Mapping[str, Any]]] = None,
) -> float:
    """
    過剰取引（高頻度取引）に対する正規化されたペナルティを返します。

    1日あたりの平均取引回数が増えるにつれてペナルティは増加し、
    双曲線正接（tanh）により ``[0, 1)`` の範囲に収まります。

    Args:
        total_trades: 総取引回数。
        start_date: 開始日時。
        end_date: 終了日時。
        trade_history: 取引履歴（total_tradesが0の場合にカウントに使用）。

    Returns:
        ペナルティ値（0.0〜1.0未満）。
    """

    trades = int(total_trades or 0)
    if trades <= 0 and trade_history is not None:
        trades = sum(1 for _ in trade_history)

    if trades <= 0:
        return 0.0

    parsed_start = _ensure_datetime(start_date)
    parsed_end = _ensure_datetime(end_date)

    if parsed_start is None or parsed_end is None or parsed_end <= parsed_start:
        duration_days = 1.0
    else:
        duration_days = max(
            (parsed_end - parsed_start).total_seconds() / 86_400.0,
            1.0 / 24.0,
        )

    trades_per_day = trades / duration_days

    return math.tanh(trades_per_day / REFERENCE_TRADES_PER_DAY)
