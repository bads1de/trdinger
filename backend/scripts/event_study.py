"""イベントスタディ実行スクリプト(凍結パラメータ版).

docs/event_study_research_plan.md の H1-H5 を、事前登録した凍結パラメータの
まま実行する。判定は確認サンプル(BTC 2024-2026 + ETH全体)のみで行う。

注意(実装上のサンプリング規約・判定基準への追加ではない):
- 同一仮説のイベントが8h以内に連続する場合は最初の1件のみ採用(独立性確保)
- フォワードリターンは1h終値ベース。窓は 8/24/72/168h
- 重複窓による n の水増しがあるため、n は参考値として注記する

Usage:
    uv run python -m scripts.event_study
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ===== 凍結パラメータ(docs/event_study_research_plan.md §3) =====
WINDOWS_H = (8, 24, 72, 168)
PRIMARY_WINDOW_H = 24
FR_TRAILING_DAYS = 90
FR_PCT = 2.0
FR_ABS_THRESHOLD = 0.0005  # ±0.05%(FRは少数値: 0.0005=0.05%)
OI_FLUSH_24H = -0.15
LSR_TRAILING_DAYS = 30
LSR_PCT = 5.0
EVENT_MIN_SPACING_H = 8
CONFIRM_SYMBOLS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
CONFIRM_BTC_START = "2024-01-01"
PASS_MEDIAN_ABOVE_BASELINE = True
PASS_WIN_RATE = 52.0
PASS_MIN_N = 100


def _naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_localize(None) if idx.tz is not None else idx


def load_hourly_closes(symbol: str) -> pd.Series:
    from database.connection import SessionLocal
    from database.repositories.ohlcv_repository import OHLCVRepository

    db = SessionLocal()
    try:
        repo = OHLCVRepository(db)
        df = repo.get_ohlcv_dataframe(symbol, "1h")
    finally:
        db.close()
    idx = _naive(pd.DatetimeIndex(df.index))
    return pd.Series(df["close"].astype(float).values, index=idx).sort_index()


def load_funding(symbol: str) -> pd.Series:
    from database.connection import SessionLocal
    from database.models import FundingRateData

    db = SessionLocal()
    try:
        rows = (
            db.query(FundingRateData.funding_timestamp, FundingRateData.funding_rate)
            .filter(FundingRateData.symbol == symbol)
            .order_by(FundingRateData.funding_timestamp)
            .all()
        )
    finally:
        db.close()
    return pd.Series(
        [float(r[1]) for r in rows],
        index=_naive(pd.DatetimeIndex([r[0] for r in rows])),
    )


def load_open_interest(symbol: str) -> pd.Series:
    from database.connection import SessionLocal
    from database.models import OpenInterestData

    db = SessionLocal()
    try:
        rows = (
            db.query(
                OpenInterestData.data_timestamp, OpenInterestData.open_interest_value
            )
            .filter(OpenInterestData.symbol == symbol)
            .order_by(OpenInterestData.data_timestamp)
            .all()
        )
    finally:
        db.close()
    return pd.Series(
        [float(r[1]) for r in rows],
        index=_naive(pd.DatetimeIndex([r[0] for r in rows])),
    )


def load_lsr(symbol: str, period: str = "1h") -> pd.Series:
    from database.connection import SessionLocal
    from database.models import LongShortRatioData

    db = SessionLocal()
    try:
        rows = (
            db.query(LongShortRatioData.timestamp, LongShortRatioData.buy_ratio)
            .filter(
                LongShortRatioData.symbol == symbol,
                LongShortRatioData.period == period,
            )
            .order_by(LongShortRatioData.timestamp)
            .all()
        )
    finally:
        db.close()
    return pd.Series(
        [float(r[1]) for r in rows],
        index=_naive(pd.DatetimeIndex([r[0] for r in rows])),
    )


# ===== 純粋関数(テスト対象) =====


def trailing_percentile_flags(
    series: pd.Series, trailing_days: int, pct: float
) -> tuple[pd.Series, pd.Series]:
    """各時点の「直後trailing_days日の履歴におけるパーセンタイル」で極端フラグを返す。

    Returns: (low_flag, high_flag)。履歴不足の時点は False。
    min_obs は系列の実際の間隔から推定する(FR=8h, LSR/OI=1h 等に対応)。
    """
    if len(series) < 2:
        empty = pd.Series(False, index=series.index)
        return empty, empty
    step_hours = float(
        series.index.to_series().diff().dt.total_seconds().div(3600).median()
    )
    if step_hours <= 0 or pd.isna(step_hours):
        step_hours = 1.0
    expected_rows = int(trailing_days * 24 / step_hours)
    min_obs = max(10, int(expected_rows * 0.9))
    roll = series.rolling(f"{trailing_days}D", min_periods=min_obs)
    rank_pct = roll.rank(pct=True)
    low = rank_pct <= (pct / 100.0)
    high = rank_pct >= (1.0 - pct / 100.0)
    valid = rank_pct.notna()
    return (low & valid), (high & valid)


def detect_fr_extremes(fr: pd.Series) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """H1/H2: 直近90日で下位2%かつ<-0.05%(H1)/上位2%かつ>+0.05%(H2)。"""
    if fr.empty:
        return fr.index[:0], fr.index[:0]
    low, high = trailing_percentile_flags(fr, FR_TRAILING_DAYS, FR_PCT)
    h1 = low & (fr < FR_ABS_THRESHOLD)
    h2 = high & (fr > abs(FR_ABS_THRESHOLD))
    return fr.index[h1.fillna(False)], fr.index[h2.fillna(False)]


def detect_oi_flush(oi: pd.Series) -> pd.DatetimeIndex:
    """H3: 24h OI変化 ≤ −15%。"""
    if oi.empty:
        return oi.index[:0]
    chg = oi / oi.shift(24) - 1.0
    mask = (chg <= OI_FLUSH_24H).fillna(False)
    return oi.index[mask]


def detect_lsr_extremes(lsr: pd.Series) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """H4: LSR(buy_ratio)の30日rollingパーセンタイル 下位5%/上位5%。"""
    if lsr.empty:
        return lsr.index[:0], lsr.index[:0]
    low, high = trailing_percentile_flags(lsr, LSR_TRAILING_DAYS, LSR_PCT)
    return lsr.index[low.fillna(False)], lsr.index[high.fillna(False)]


def dedup_events(times: pd.DatetimeIndex, min_spacing_h: int) -> pd.DatetimeIndex:
    """min_spacing_h 以内の連続イベントを間引く(先着優先)。"""
    if len(times) == 0:
        return times
    kept: list[pd.Timestamp] = []
    threshold = pd.Timedelta(hours=min_spacing_h)
    for ts in times:
        if not kept or ts - kept[-1] >= threshold:
            kept.append(ts)
    return pd.DatetimeIndex(kept)


def forward_returns(
    closes_hourly: pd.Series, windows_h: tuple[int, ...]
) -> dict[int, pd.Series]:
    """各時点からのフォワードリターン(小数)。データ末尾の未完結窓は NaN。"""
    out: dict[int, pd.Series] = {}
    for w in windows_h:
        out[w] = closes_hourly.shift(-w) / closes_hourly - 1.0
    return out


def align_events_to_hours(
    times: pd.DatetimeIndex, hourly_index: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """イベント時刻を「その直前の1hバー終値時刻」へ写像。対応バー無しは除外。"""
    pos = hourly_index.searchsorted(times, side="right") - 1
    valid = pos >= 0
    mapped = hourly_index[pos[valid]]
    return pd.DatetimeIndex(mapped)


def summarize_cell(
    event_hours: pd.DatetimeIndex,
    fwd: pd.Series,
    baseline_median: float,
) -> dict[str, Any] | None:
    vals = fwd.reindex(event_hours).dropna()
    if len(vals) == 0:
        return None
    pp = vals * 100.0
    wins = int((pp > 0).sum())
    return {
        "n": int(len(pp)),
        "median_pp": round(float(pp.median()), 4),
        "mean_pp": round(float(pp.mean()), 4),
        "baseline_median_pp": round(baseline_median * 100.0, 4),
        "edge_pp": round((float(pp.median()) - baseline_median * 100.0), 4),
        "win_rate": round(wins / len(pp) * 100.0, 2),
    }


def in_confirm_sample(ts: pd.Timestamp, symbol: str) -> bool:
    if symbol not in CONFIRM_SYMBOLS:
        return False
    if symbol.startswith("BTC"):
        return bool(ts >= pd.Timestamp(CONFIRM_BTC_START))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="事前登録済みイベントスタディ(H1-H5)")
    parser.add_argument("--out", default="results/event_study_results.jsonl")
    parser.add_argument(
        "--symbols",
        default=None,
        help="対象シンボル(カンマ区切り)。未指定時はCONFIRM_SYMBOLS",
    )
    parser.add_argument(
        "--hyps",
        default=None,
        help="実行する仮説(カンマ区切り、例: H3_OI_FLUSH)。未指定時は全仮説",
    )
    args = parser.parse_args()
    symbols = (
        tuple(s.strip() for s in args.symbols.split(",") if s.strip())
        if args.symbols
        else CONFIRM_SYMBOLS
    )
    hyp_filter = (
        {s.strip() for s in args.hyps.split(",") if s.strip()} if args.hyps else None
    )

    all_rows: list[dict[str, Any]] = []
    for sym in symbols:
        closes = load_hourly_closes(sym)
        fwd = forward_returns(closes, WINDOWS_H)
        baselines = {w: float(s.dropna().median()) for w, s in fwd.items()}

        fr = load_funding(sym)
        h1_ts, h2_ts = detect_fr_extremes(fr)
        oi = load_open_interest(sym)
        h3_ts = detect_oi_flush(oi)
        lsr = load_lsr(sym)
        h4_low_ts, h4_high_ts = detect_lsr_extremes(lsr)

        conf_raw = sorted(set(h1_ts) | set(h4_low_ts))
        conf_ts = dedup_events(pd.DatetimeIndex(conf_raw), EVENT_MIN_SPACING_H)

        events: dict[str, pd.DatetimeIndex] = {
            "H1_FR_LOW": dedup_events(h1_ts, EVENT_MIN_SPACING_H),
            "H2_FR_HIGH": dedup_events(h2_ts, EVENT_MIN_SPACING_H),
            "H3_OI_FLUSH": dedup_events(h3_ts, EVENT_MIN_SPACING_H),
            "H4_LSR_LOW": dedup_events(h4_low_ts, EVENT_MIN_SPACING_H),
            "H4_LSR_HIGH": dedup_events(h4_high_ts, EVENT_MIN_SPACING_H),
            "H5_CONF": dedup_events(conf_ts, EVENT_MIN_SPACING_H),
        }
        if hyp_filter is not None:
            events = {k: v for k, v in events.items() if k in hyp_filter}

        print(f"\n===== {sym} =====")
        for hyp, ts in events.items():
            hours_all = align_events_to_hours(ts, closes.index)
            hours_conf = pd.DatetimeIndex(
                [t for t in hours_all if in_confirm_sample(t, sym)]
            )
            for w in WINDOWS_H:
                for sample_name, ev_hours in (
                    ("full", hours_all),
                    ("confirm", hours_conf),
                ):
                    cell = summarize_cell(ev_hours, fwd[w], baselines[w])
                    if cell is None:
                        continue
                    row = {
                        "symbol": sym,
                        "hyp": hyp,
                        "window_h": w,
                        "sample": sample_name,
                        **cell,
                    }
                    all_rows.append(row)
            c24 = next(
                (
                    r
                    for r in reversed(all_rows)
                    if r["hyp"] == hyp
                    and r["window_h"] == PRIMARY_WINDOW_H
                    and r["sample"] == "confirm"
                    and r["symbol"] == sym
                ),
                None,
            ) or next(
                (
                    r
                    for r in reversed(all_rows)
                    if r["hyp"] == hyp
                    and r["window_h"] == PRIMARY_WINDOW_H
                    and r["sample"] == "full"
                    and r["symbol"] == sym
                ),
                None,
            )
            if c24 is None:
                print(f"{hyp:12s} 24h: イベントなし")
                continue
            print(
                f"{hyp:12s} 確認24h: n={c24['n']:>4d} 中央値 {c24['median_pp']:+7.3f}pp "
                f"(基準 {c24['baseline_median_pp']:+7.3f}) edge {c24['edge_pp']:+7.3f}pp "
                f"勝率 {c24['win_rate']}%"
            )

    out_path = PROJECT_ROOT / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n保存: {out_path} ({len(all_rows)} cells)")


if __name__ == "__main__":
    main()
