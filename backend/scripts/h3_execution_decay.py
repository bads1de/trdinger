"""H3執行タイミング減衰分析(docs/event_study_research_plan.md §9 残存課題a+b).

凍結イベント定義(H3: 24h OI変化 ≤ −15%, 8hデデュープ)はそのままに、
検出からエントリーを d 時間遅延させた場合の24hフォワードリターンを測定する。
記述統計であり合否判定は行わない。

併せて以下を報告する:
- ホールディング曲線(d=0でエントリーし 2/4/8/12/24/48/72h 保持)
- エピソードクラスタリング(72h以内の連続イベントを同一エピソードとみなす)

Usage:
    uv run python -m scripts.h3_execution_decay
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
    detect_oi_flush,
    load_hourly_closes,
    load_open_interest,
)

SYMBOLS = tuple(f"{s}/USDT:USDT" for s in ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE"))
DELAY_H = (0, 1, 2, 4, 8, 12)
HOLDING_H = (2, 4, 8, 12, 24, 48, 72)
EPISODE_GAP_H = 72
ROUND_TRIP_COST_PP = 0.08


def delayed_forward_returns(
    closes_hourly: pd.Series, delay_h: int, horizon_h: int
) -> pd.Series:
    """各時点 t について「t+delay_h 終値でエントリーし t+delay_h+horizon_h 終値で
    エグジットした場合のリターン(小数)」を返す。末尾の未完結窓は NaN。"""
    if delay_h < 0 or horizon_h <= 0:
        raise ValueError("delay_h>=0, horizon_h>0 が必要")
    entry = closes_hourly.shift(-delay_h)
    exit_ = closes_hourly.shift(-(delay_h + horizon_h))
    return exit_ / entry - 1.0


def count_episodes(event_hours: pd.DatetimeIndex, gap_hours: int) -> int:
    """gap_hours 以上空いたところで区切ったエピソード数を数える。"""
    if len(event_hours) == 0:
        return 0
    gaps = event_hours.to_series().diff().dt.total_seconds().div(3600)
    starts = (gaps >= gap_hours) | gaps.isna()
    return int(starts.sum())


def summarize(vals: pd.Series) -> dict[str, Any] | None:
    v = vals.dropna()
    if len(v) == 0:
        return None
    pp = v * 100.0
    return {
        "n": int(len(pp)),
        "median_pp": round(float(pp.median()), 4),
        "win_rate": round(float((pp > 0).mean()) * 100.0, 2),
    }


def main() -> None:
    out_rows: list[dict[str, Any]] = []
    pooled_delays: dict[int, list[pd.Series]] = {d: [] for d in DELAY_H}
    pooled_hold: dict[int, list[pd.Series]] = {w: [] for w in HOLDING_H}

    for sym in SYMBOLS:
        closes = load_hourly_closes(sym)
        oi = load_open_interest(sym)
        events = dedup_events(detect_oi_flush(oi), EVENT_MIN_SPACING_H)
        ev_hours = align_events_to_hours(events, closes.index)

        print(
            f"\n===== {sym} =====  events={len(ev_hours)} "
            f"episodes={count_episodes(ev_hours, EPISODE_GAP_H)}"
        )

        for d in DELAY_H:
            fwd = delayed_forward_returns(closes, d, 24)
            s = summarize(fwd.reindex(ev_hours))
            b = summarize(fwd.dropna())
            if s is None or b is None:
                print(f"  delay={d:>2}h: イベントなし")
                continue
            edge = round(s["median_pp"] - b["median_pp"], 4)
            net = round(s["median_pp"] - ROUND_TRIP_COST_PP, 4)
            print(
                f"  delay={d:>2}h n={s['n']:>4} 中央値 {s['median_pp']:+7.3f}pp "
                f"(全時間基準 {b['median_pp']:+6.3f}) edge {edge:+7.3f}pp "
                f"net-of-cost {net:+7.3f}pp 勝率 {s['win_rate']}%"
            )
            out_rows.append(
                {
                    "symbol": sym,
                    "kind": "delay",
                    "param_h": d,
                    **s,
                    "all_time_median_pp": b["median_pp"],
                    "edge_pp": edge,
                }
            )
            pooled_delays[d].append(fwd.reindex(ev_hours))

        for w in HOLDING_H:
            fwd = delayed_forward_returns(closes, 0, w)
            s = summarize(fwd.reindex(ev_hours))
            b = summarize(fwd.dropna())
            if s is None or b is None:
                continue
            print(
                f"  hold ={w:>2}h n={s['n']:>4} 中央値 {s['median_pp']:+7.3f}pp "
                f"(基準 {b['median_pp']:+6.3f}) 勝率 {s['win_rate']}%"
            )
            out_rows.append(
                {
                    "symbol": sym,
                    "kind": "hold",
                    "param_h": w,
                    **s,
                    "all_time_median_pp": b["median_pp"],
                }
            )
            pooled_hold[w].append(fwd.reindex(ev_hours))

    print("\n===== プール(6銘柄) =====")
    print("-- エントリー遅延 vs 24h保持 --")
    for d in DELAY_H:
        vals = pd.concat(pooled_delays[d])
        s = summarize(vals)
        if s is None:
            continue
        print(
            f"  delay={d:>2}h n={s['n']:>5} 中央値 {s['median_pp']:+7.3f}pp "
            f"net {s['median_pp'] - ROUND_TRIP_COST_PP:+7.3f}pp 勝率 {s['win_rate']}%"
        )
    print("-- ホールディング曲線(d=0エントリー) --")
    for w in HOLDING_H:
        vals = pd.concat(pooled_hold[w])
        s = summarize(vals)
        if s is None:
            continue
        print(
            f"  hold ={w:>2}h n={s['n']:>5} 中央値 {s['median_pp']:+7.3f}pp 勝率 {s['win_rate']}%"
        )

    out_path = PROJECT_ROOT / "results" / "h3_execution_decay.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n保存: {out_path} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
