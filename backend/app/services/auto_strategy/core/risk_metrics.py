"""Risk-focused fitness metric helpers."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Mapping, MutableSequence, Optional, Sequence


REFERENCE_TRADES_PER_DAY = 8.0


def calculate_ulcer_index(equity_curve: Sequence[Mapping[str, Any]]) -> float:
    """Compute the ulcer index (RMS drawdown).

    Args:
        equity_curve: Iterable of equity points produced by the backtest converter.

    Returns:
        Root mean square of drawdown percentages expressed as decimals.
    """

    if not equity_curve:
        return 0.0

    squared_drawdowns: MutableSequence[float] = []
    for point in equity_curve:
        raw_drawdown: Any = point.get("drawdown") if isinstance(point, Mapping) else None
        if raw_drawdown is None:
            continue
        try:
            drawdown = float(raw_drawdown)
        except (TypeError, ValueError):
            continue

        if math.isnan(drawdown):
            continue

        drawdown = abs(drawdown)
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
    """Return a normalized penalty for excessive trade frequency.

    The penalty increases with the average number of trades per day and is bounded
    in ``[0, 1)`` via a hyperbolic tangent.
    """

    trades = int(total_trades or 0)
    if trades <= 0 and trade_history is not None:
        trades = sum(1 for _ in trade_history)

    if trades <= 0:
        return 0.0

    parsed_start = _ensure_datetime(start_date)
    parsed_end = _ensure_datetime(end_date)

    if parsed_start is None or parsed_end is None or parsed_end <= parsed_start:
        # フォールバックで1日とみなす
        duration_days = 1.0
    else:
        duration_days = max(
            (parsed_end - parsed_start).total_seconds() / 86_400.0,
            1.0 / 24.0,
        )

    trades_per_day = trades / duration_days

    return math.tanh(trades_per_day / REFERENCE_TRADES_PER_DAY)


def _ensure_datetime(value: Optional[object]) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None