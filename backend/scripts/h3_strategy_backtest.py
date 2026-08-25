"""H3戦略ポートフォリオバックテスト(docs/event_study_research_plan.md §12凍結スペック).

シグナル: 24h OI変化 ≤ −15%(銘柄内8hデデュープ)
エントリー: 検出バーの翌バー始値 / イグジット: エントリー24h後のバー始値
サイジング: エントリー時点の資本の10% / 同時保有上限3(先着優先)
コスト: 往復8bp標準・20bpストレス。単一実行(パラメータ反復禁止)。

Usage:
    uv run python -m scripts.h3_strategy_backtest
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from scripts.event_study import (
    EVENT_MIN_SPACING_H,
    PROJECT_ROOT,
    align_events_to_hours,
    dedup_events,
    detect_oi_flush,
    load_open_interest,
)
from scripts.h3_execution_decay import SYMBOLS

EQUITY_START = 10_000.0
SIZE_FRACTION = 0.10
MAX_CONCURRENT = 3
HOLD_H = 24
COST_STANDARD = 0.0008
COST_STRESS = 0.0020


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_px: float
    exit_px: float
    notional: float

    @property
    def ret(self) -> float:
        return self.exit_px / self.entry_px - 1.0


def _naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_localize(None) if idx.tz is not None else idx


def load_prices(symbol: str) -> tuple[pd.Series, pd.Series]:
    """(始値, 終値)の1h系列を返す。"""
    from database.connection import SessionLocal
    from database.repositories.ohlcv_repository import OHLCVRepository

    db = SessionLocal()
    try:
        repo = OHLCVRepository(db)
        df = repo.get_ohlcv_dataframe(symbol, "1h")
    finally:
        db.close()
    idx = _naive(pd.DatetimeIndex(df.index))
    opens = pd.Series(df["open"].astype(float).values, index=idx).sort_index()
    closes = pd.Series(df["close"].astype(float).values, index=idx).sort_index()
    return opens, closes


def collect_signals(
    events_by_symbol: dict[str, pd.DatetimeIndex],
    opens_by_symbol: dict[str, pd.Series],
) -> list[tuple[pd.Timestamp, str]]:
    """全シンボルの(エントリー時刻=検出翌バー, シンボル)を時刻順で返す。"""
    out: list[tuple[pd.Timestamp, str]] = []
    for sym, ev in events_by_symbol.items():
        idx = opens_by_symbol[sym].index
        for t in align_events_to_hours(ev, idx):
            i = int(idx.get_loc(t))
            if i + 1 < len(idx):
                out.append((idx[i + 1], sym))
    return sorted(out)


def plan_exit(entry_ts: pd.Timestamp, oidx: pd.DatetimeIndex) -> int | None:
    """エントリーから24h後以降で最初のバー位置。無ければ None。"""
    xpos = int(oidx.searchsorted(entry_ts + pd.Timedelta(hours=HOLD_H)))
    return xpos if xpos < len(oidx) else None


def execute_portfolio(
    signals: list[tuple[pd.Timestamp, str]],
    opens_by_symbol: dict[str, pd.Series],
    cost_rt: float,
) -> tuple[list[Trade], list[pd.Timestamp], list[float]]:
    """同時保有上限3・資本10%固定サイズでトレードを実行する。

    Returns: (トレード群, スキップ時刻群, 各エントリー時点の資本)
    """
    executed: list[Trade] = []
    skipped: list[pd.Timestamp] = []
    equity_at_entry: list[float] = []
    cash = EQUITY_START
    active: list[Trade] = []

    for sig_ts, sym in signals:
        active = [p for p in active if p.exit_ts > sig_ts]
        oidx = opens_by_symbol[sym].index
        epos = int(oidx.searchsorted(sig_ts))
        if epos >= len(oidx):
            continue
        xpos = plan_exit(sig_ts, oidx)
        if xpos is None:
            continue
        if len(active) >= MAX_CONCURRENT:
            skipped.append(sig_ts)
            continue
        equity_now = cash + sum(p.notional for p in active)
        allocated = SIZE_FRACTION * equity_now
        trade = Trade(
            symbol=sym,
            entry_ts=sig_ts,
            exit_ts=oidx[xpos],
            entry_px=float(opens_by_symbol[sym].iloc[epos]),
            exit_px=float(opens_by_symbol[sym].iloc[xpos]),
            notional=allocated,
        )
        pnl = allocated * (trade.ret - cost_rt)
        executed.append(trade)
        equity_at_entry.append(equity_now)
        cash += pnl
        active.append(trade)
    return executed, skipped, equity_at_entry


def mark_equity(
    trades: list[Trade],
    closes_by_symbol: dict[str, pd.Series],
    cost_rt: float,
) -> pd.Series:
    """1hグリッド上のMTM資本曲線。"""
    master = pd.DatetimeIndex(
        sorted(set().union(*(set(c.index) for c in closes_by_symbol.values())))
    )
    cash_step = pd.Series(0.0, index=master)
    pos_val = pd.Series(0.0, index=master)
    for tr in trades:
        cash_step.loc[tr.exit_ts] += tr.notional * (tr.ret - cost_rt)
        cs = closes_by_symbol[tr.symbol]
        seg_idx = master[(master >= tr.entry_ts) & (master <= tr.exit_ts)]
        px = cs.reindex(seg_idx, method="ffill")
        pos_val.loc[seg_idx] += tr.notional * px / tr.entry_px
    cash = EQUITY_START + cash_step.cumsum()
    eq = cash + pos_val
    return eq


def perf_metrics(eq: pd.Series, trades: list[Trade]) -> dict[str, Any]:
    r = eq.pct_change().dropna()
    hours_per_year = 24 * 365
    sharpe = (
        float(r.mean() / r.std() * math.sqrt(hours_per_year)) if r.std() > 0 else 0.0
    )
    dd = float((eq / eq.cummax() - 1.0).min())
    years = max((eq.index[-1] - eq.index[0]).total_seconds() / (365 * 24 * 3600), 1e-9)
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    wins = [t for t in trades if t.ret > 0]
    gp = sum(t.notional * t.ret for t in wins)
    gl = abs(sum(t.notional * t.ret for t in trades if t.ret <= 0))
    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(((1 + total_ret) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(dd * 100, 2),
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else float("inf"),
        "avg_trade_pp": round(sum(t.ret for t in trades) / len(trades) * 100, 4)
        if trades
        else 0.0,
    }


def main() -> None:
    prices = {s: load_prices(s) for s in SYMBOLS}
    opens_by_symbol = {s: p[0] for s, p in prices.items()}
    closes_by_symbol = {s: p[1] for s, p in prices.items()}
    oi_by_symbol = {s: load_open_interest(s) for s in SYMBOLS}

    events_by_symbol = {
        s: dedup_events(detect_oi_flush(o), EVENT_MIN_SPACING_H)
        for s, o in oi_by_symbol.items()
    }
    n_ev = sum(len(v) for v in events_by_symbol.values())
    print(f"イベント総数(デデュープ後): {n_ev}")
    signals = collect_signals(events_by_symbol, opens_by_symbol)

    results: dict[str, Any] = {}
    rows_out: list[dict[str, Any]] = []
    for label, cost_bp in (("standard", COST_STANDARD), ("stress20", COST_STRESS)):
        trades, skipped, _ = execute_portfolio(signals, opens_by_symbol, cost_bp)
        eq = mark_equity(trades, closes_by_symbol, cost_bp)
        m = perf_metrics(eq, trades)
        m["skipped_signals"] = len(skipped)
        by_sym: dict[str, dict[str, Any]] = {}
        for s in SYMBOLS:
            st = [t for t in trades if t.symbol == s]
            by_sym[s.split("/")[0]] = {
                "n": len(st),
                "win_rate": round(len([t for t in st if t.ret > 0]) / len(st) * 100, 2)
                if st
                else 0.0,
                "pnl_usd": round(sum(t.notional * (t.ret - cost_bp) for t in st), 2),
            }
        m["by_symbol"] = by_sym
        results[label] = m
        print(f"\n===== {label} (往復{cost_bp:.2%}) =====")
        for k, v in m.items():
            if k != "by_symbol":
                print(f"  {k}: {v}")
        for s, d in by_sym.items():
            print(f"    {s}: n={d['n']} win={d['win_rate']}% pnl=${d['pnl_usd']}")
        for t in trades:
            rows_out.append(
                {
                    "cost_scenario": label,
                    "symbol": t.symbol,
                    "entry_ts": str(t.entry_ts),
                    "exit_ts": str(t.exit_ts),
                    "ret_pct": round((t.ret - cost_bp) * 100, 4),
                    "notional": round(t.notional, 2),
                    "pnl": round(t.notional * (t.ret - cost_bp), 4),
                }
            )

    gates = {
        "sharpe>=0.7": results["standard"]["sharpe"] >= 0.7,
        "pf>=1.15": results["standard"]["profit_factor"] >= 1.15,
        "maxdd<25%": results["standard"]["max_drawdown_pct"] > -25.0,
        "positive_symbols>=4/6": len(
            [d for d in results["standard"]["by_symbol"].values() if d["pnl_usd"] > 0]
        )
        >= 4,
    }
    print("\n===== 合格基準(§12) =====")
    for g, ok in gates.items():
        print(f"  {g}: {'PASS' if ok else 'FAIL'}")

    out_path = PROJECT_ROOT / "results" / "h3_strategy_backtest.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = PROJECT_ROOT / "results" / "h3_strategy_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "gates": gates}, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {out_path} ({len(rows_out)} rows), {summary_path}")


if __name__ == "__main__":
    main()
