"""H3閾値感度分析(docs/event_study_research_plan.md §9残存課題c).

凍結閾値−15%の前後でイベント深度を動かし、エッジの用量反応
(より深い清算フラッシュほど強い反発)を確認する。記述統計。

条件: 24h保持・即時エントリー・8hデデュープ・6銘柄プール。

Usage:
    uv run python -m scripts.h3_threshold_sensitivity
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from scripts.event_study import (
    EVENT_MIN_SPACING_H,
    PROJECT_ROOT,
    align_events_to_hours,
    dedup_events,
    load_hourly_closes,
    load_open_interest,
)
from scripts.h3_execution_decay import (
    SYMBOLS,
    count_episodes,
    delayed_forward_returns,
    summarize,
)

THRESHOLDS = (-0.10, -0.125, -0.15, -0.20, -0.25)
HORIZON_H = 24


def detect_oi_flush_at(oi: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """24h OI変化が threshold 以下の時刻を返す(H3の凍結定義の閾値のみ変更版)。"""
    if oi.empty:
        return oi.index[:0]
    chg = oi / oi.shift(24) - 1.0
    mask = (chg <= threshold).fillna(False)
    return oi.index[mask]


def main() -> None:
    out_rows: list[dict[str, Any]] = []
    closes_map = {sym: load_hourly_closes(sym) for sym in SYMBOLS}
    fwd_map = {
        sym: delayed_forward_returns(closes_map[sym], 0, HORIZON_H) for sym in SYMBOLS
    }
    base_map = {sym: float(fwd.dropna().median()) for sym, fwd in fwd_map.items()}

    for th in THRESHOLDS:
        print(f"\n===== 閾値 {th:+.1%} =====")
        pooled_vals: list[pd.Series] = []
        total_ep = 0
        for sym in SYMBOLS:
            ev = align_events_to_hours(
                dedup_events(
                    detect_oi_flush_at(load_open_interest(sym), th),
                    EVENT_MIN_SPACING_H,
                ),
                closes_map[sym].index,
            )
            s = summarize(fwd_map[sym].reindex(ev))
            ep = count_episodes(ev, 72)
            total_ep += ep
            if s is None:
                continue
            print(
                f"  {sym.split('/')[0]:>4} n={s['n']:>4} 中央値 {s['median_pp']:+7.3f}pp "
                f"(基準 {base_map[sym] * 100:+6.3f}) 勝率 {s['win_rate']}% ep={ep}"
            )
            out_rows.append(
                {
                    "threshold": th,
                    "symbol": sym,
                    **s,
                    "baseline_median_pp": round(base_map[sym] * 100, 4),
                    "episodes": ep,
                }
            )
            pooled_vals.append(fwd_map[sym].reindex(ev))
        vals = pd.concat(pooled_vals)
        p = summarize(vals)
        if p is not None:
            bl_avg = sum(base_map.values()) / len(base_map) * 100
            net = round(p["median_pp"] - 0.08, 4)
            print(
                f"  プール n={p['n']} 中央値 {p['median_pp']:+7.3f}pp "
                f"(基準平均 {bl_avg:+6.3f}) net-of-cost {net:+7.3f}pp "
                f"勝率 {p['win_rate']}% episodes={total_ep}"
            )
            out_rows.append(
                {
                    "threshold": th,
                    "symbol": "POOLED",
                    **p,
                    "baseline_median_pp": round(bl_avg, 4),
                    "net_of_cost_median_pp": net,
                    "episodes": total_ep,
                }
            )

    out_path = PROJECT_ROOT / "results" / "h3_threshold_sensitivity.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n保存: {out_path} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
