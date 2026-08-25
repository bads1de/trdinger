import pandas as pd
import pytest

from scripts.h3_strategy_backtest import (
    EQUITY_START,
    MAX_CONCURRENT,
    SIZE_FRACTION,
    Trade,
    collect_signals,
    execute_portfolio,
    mark_equity,
    plan_exit,
)


def _grid(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1h")


class TestPlanExit:
    def test_returns_bar_at_or_after_24h(self) -> None:
        idx = _grid("2024-01-01", 40)
        assert plan_exit(idx[0], idx) == 24

    def test_none_when_history_too_short(self) -> None:
        idx = _grid("2024-01-01", 10)
        assert plan_exit(idx[0], idx) is None


class TestCollectSignals:
    def test_next_bar_entry_ordering(self) -> None:
        opens_a = pd.Series(1.0, index=_grid("2024-01-01", 10))
        opens_b = pd.Series(1.0, index=_grid("2024-01-02", 10))
        ev = {"A": pd.DatetimeIndex([opens_a.index[2]]), "B": pd.DatetimeIndex([])}
        sig = collect_signals(ev, {"A": opens_a, "B": opens_b})
        assert sig == [(opens_a.index[3], "A")]

    def test_event_at_last_bar_dropped(self) -> None:
        opens_a = pd.Series(1.0, index=_grid("2024-01-01", 10))
        ev = {"A": pd.DatetimeIndex([opens_a.index[9]])}
        assert collect_signals(ev, {"A": opens_a}) == []


def _flat_prices(n: int, px: float = 100.0) -> tuple[pd.Series, pd.Series]:
    idx = _grid("2024-01-01", n)
    s = pd.Series(px, index=idx)
    return s.copy(), s.copy()


class TestExecutePortfolio:
    def test_sizing_and_pnl_accounting(self) -> None:
        idx = _grid("2024-01-01", 60)
        opens = pd.Series([100.0] * len(idx), index=idx)
        closes = opens.copy()
        closes.iloc[30] = 110.0  # exit bar close irrelevant; exit uses open
        signals = [(idx[1], "A")]
        trades, skipped, eqs = execute_portfolio(signals, {"A": opens}, cost_rt=0.0008)
        assert len(trades) == 1 and not skipped
        t = trades[0]
        assert t.entry_px == 100.0 and t.exit_px == 100.0
        assert t.notional == pytest.approx(SIZE_FRACTION * EQUITY_START)
        assert eqs[0] == pytest.approx(EQUITY_START)

    def test_concurrency_cap_skips_excess(self) -> None:
        idx = _grid("2024-01-01", 120)
        opens = {s: pd.Series(100.0, index=idx) for s in ("A", "B", "C", "D")}
        # 4シンボルが同一時刻にシグナル -> 上限3で1件スキップ
        signals = [(idx[1], s) for s in ("A", "B", "C", "D")]
        trades, skipped, _ = execute_portfolio(signals, opens, cost_rt=0.0008)
        assert len(trades) == MAX_CONCURRENT
        assert len(skipped) == 1

    def test_slot_freed_after_exit(self) -> None:
        idx = _grid("2024-01-01", 200)
        opens = pd.Series(100.0, index=idx)
        signals = [
            (idx[1], "A"),
            (idx[1] + pd.Timedelta(hours=24), "A"),
            (idx[1] + pd.Timedelta(hours=48), "A"),
        ]
        trades, skipped, _ = execute_portfolio(signals, {"A": opens}, 0.0008)
        # 各エントリー時点で直前トレードは既にexit済み(exact時刻解放)なので全実行
        assert len(trades) == 3 and not skipped


class TestMarkEquity:
    def test_final_equals_cash_plus_zero_open(self) -> None:
        idx = _grid("2024-01-01", 60)
        opens = pd.Series([100.0] * len(idx), index=idx)
        closes = opens.copy()
        closes.iloc[:26] = 90.0  # 保持中に下落しても最終資本は実現PnLで決まる
        signals = [(idx[1], "A")]
        trades, _, _ = execute_portfolio(signals, {"A": opens}, 0.0008)
        eq = mark_equity(trades, {"A": closes}, 0.0008)
        expected = EQUITY_START + sum(t.notional * (t.ret - 0.0008) for t in trades)
        assert eq.iloc[-1] == pytest.approx(expected)

    def test_trade_dataclass_ret(self) -> None:
        t = Trade("A", idx_ts := pd.Timestamp("2024-01-01"), idx_ts, 100.0, 101.0, 10.0)
        assert t.ret == pytest.approx(0.01)
