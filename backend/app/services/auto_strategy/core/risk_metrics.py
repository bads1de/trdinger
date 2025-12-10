"""
リスク評価のためのフィットネス指標ヘルパー関数群。
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Mapping, MutableSequence, Optional, Sequence

# 基準となる1日あたりの取引回数（これを超えると過剰取引とみなすペナルティが増加）
REFERENCE_TRADES_PER_DAY = 8.0


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

    squared_drawdowns: MutableSequence[float] = []
    for point in equity_curve:
        raw_drawdown: Any = (
            point.get("drawdown") if isinstance(point, Mapping) else None
        )
        if raw_drawdown is None:
            continue
        try:
            drawdown = float(raw_drawdown)
        except (TypeError, ValueError):
            continue

        if math.isnan(drawdown):
            continue

        drawdown = abs(drawdown)
        # ドローダウンがパーセンテージ（>1.0）の場合、小数（0.0-1.0）に変換
        if drawdown > 1.0:
            drawdown /= 100.0

        squared_drawdowns.append(drawdown * drawdown)

    if not squared_drawdowns:
        return 0.0

    mean_square = sum(squared_drawdowns) / float(len(squared_drawdowns))
    return math.sqrt(mean_square)


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
        # 期間が不明または無効な場合は1日として扱う
        duration_days = 1.0
    else:
        # 最低1時間（1/24日）として計算
        duration_days = max(
            (parsed_end - parsed_start).total_seconds() / 86_400.0,
            1.0 / 24.0,
        )

    trades_per_day = trades / duration_days

    return math.tanh(trades_per_day / REFERENCE_TRADES_PER_DAY)


def _ensure_datetime(value: Optional[object]) -> Optional[datetime]:
    """値をdatetimeオブジェクトに変換します。"""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None