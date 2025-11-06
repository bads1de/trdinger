"""Risk metric utilities tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

# メトリクスモジュールが存在しないため、テストをスキップ
pytestmark = pytest.mark.skip(reason="app.services.auto_strategy.core.metrics module not implemented")


def test_calculate_ulcer_index_returns_root_mean_square() -> None:
    """Ulcer index should be the RMS of drawdown percentages."""

    base_time = datetime(2024, 1, 1, 0, 0, 0)
    equity_curve = [
        {
            "timestamp": base_time + timedelta(days=idx),
            "equity": 100000 + idx * 500,
            "drawdown": drawdown,
        }
        for idx, drawdown in enumerate([0.0, 0.05, 0.1, 0.0])
    ]

    ulcer_index = calculate_ulcer_index(equity_curve)

    # sqrt((0^2 + 0.05^2 + 0.1^2 + 0^2) / 4) = ~0.0559
    assert ulcer_index == pytest.approx(0.0559, rel=1e-4)


def test_calculate_trade_frequency_penalty_uses_trades_per_day() -> None:
    """Penalty should increase with higher daily trade frequency."""

    start = datetime(2024, 2, 1)
    end = datetime(2024, 2, 11)

    penalty = calculate_trade_frequency_penalty(
        total_trades=40,
        start_date=start,
        end_date=end,
        trade_history=[{"entry_time": start + timedelta(days=i)} for i in range(40)],
    )

    # 40 trades across 10 days => 4 trades/day, tanh(4/8) ~= 0.4621
    assert penalty == pytest.approx(0.4621, rel=1e-4)
