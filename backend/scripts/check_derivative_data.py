"""派生データ(FR/OI/LSR)の健全性チェック.

docs/event_study_research_plan.md §4 の事前チェックを実行する。
イベントスタディ(H1-H5)の実施可否判定(Gate 1)のための診断ツール。

Usage:
    uv run python -m scripts.check_derivative_data [--symbol BTC/USDT:USDT]
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED = ("BTC/USDT:USDT", "ETH/USDT:USDT")
# 既知の大規模清算カスケード日(UTC)。この日らにOI急落が再現されるか検証する。
KNOWN_CASCADE_DAYS = ("2024-08-05", "2025-10-10")
FR_EXPECTED_INTERVAL_H = 8


def _naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_localize(None) if idx.tz is not None else idx


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
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.Series([float(r[1]) for r in rows], index=_naive(idx), name="fr")


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
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.Series([float(r[1]) for r in rows], index=_naive(idx), name="oi")


def load_lsr_periods(symbol: str) -> dict[str, pd.DataFrame]:
    from database.connection import SessionLocal
    from database.models import LongShortRatioData

    db = SessionLocal()
    try:
        rows = (
            db.query(
                LongShortRatioData.period,
                LongShortRatioData.timestamp,
                LongShortRatioData.buy_ratio,
                LongShortRatioData.sell_ratio,
            )
            .filter(LongShortRatioData.symbol == symbol)
            .order_by(LongShortRatioData.timestamp)
            .all()
        )
    finally:
        db.close()
    frames: dict[str, pd.DataFrame] = {}
    if not rows:
        return frames
    df = pd.DataFrame(
        rows,
        columns=["period", "timestamp", "buy_ratio", "sell_ratio"],
    )
    df["timestamp"] = _naive(pd.DatetimeIndex(df["timestamp"]))
    for period, g in df.groupby("period"):
        frames[str(period)] = g.drop(columns=["period"]).set_index("timestamp")
    return frames


def check_funding(fr: pd.Series) -> dict[str, Any]:
    """FR系列の健全性: 欠落率・異常値・分布。"""
    out: dict[str, Any] = {"n": int(len(fr))}
    if fr.empty:
        return out
    out["range"] = [str(fr.index[0]), str(fr.index[-1])]
    gaps_hours = fr.index.to_series().diff().dt.total_seconds().div(3600)
    expected = float(FR_EXPECTED_INTERVAL_H)
    regular = gaps_hours.iloc[1:]
    n_reg = len(regular)
    missing_slots = int(((regular - expected) / expected).round().clip(lower=0).sum())
    out.update(
        {
            "gap_median_h": float(regular.median()) if n_reg else None,
            "gap_max_h": float(regular.max()) if n_reg else None,
            "missing_slot_estimate": missing_slots,
            "missing_rate_pct": round(missing_slots / max(n_reg, 1) * 100, 3),
            "outliers_gt_1pct": int((fr.abs() > 0.01).sum()),
            "pct": {
                f"p{p}": float(fr.quantile(p / 100))
                for p in (0.1, 1, 25, 50, 75, 99, 99.9)
            },
        }
    )
    return out


def check_open_interest(oi: pd.Series) -> dict[str, Any]:
    """OI系列の健全性: カバレッジ、既知カスケード日の急落再現、最大下落日。"""
    out: dict[str, Any] = {"n": int(len(oi))}
    if oi.empty:
        return out
    out["range"] = [str(oi.index[0]), str(oi.index[-1])]
    hourly = oi.resample("1h").mean().dropna()

    cascades: dict[str, Any] = {}
    for day in KNOWN_CASCADE_DAYS:
        d0 = pd.Timestamp(day)
        win = hourly[
            (hourly.index >= d0 - timedelta(days=2))
            & (hourly.index < d0 + timedelta(days=2))
        ]
        if win.empty:
            cascades[day] = "no data"
            continue
        pre_peak = float(win[win.index < d0].max())
        trough = float(win.min())
        cascades[day] = {
            "pre_48h_peak": round(pre_peak, 2),
            "window_trough": round(trough, 2),
            "drawdown_pct": round((trough / pre_peak - 1) * 100, 2)
            if pre_peak
            else None,
        }
    out["known_cascades"] = cascades

    daily_min = hourly.resample("1D").min()
    daily_max_prev = hourly.resample("1D").max().shift(1)
    dd = (daily_min / daily_max_prev - 1) * 100
    worst = dd.nsmallest(5).dropna()
    out["top5_daily_drawdowns"] = [
        {"date": str(k.date()), "drawdown_pct": round(float(v), 2)}
        for k, v in worst.items()
    ]
    gaps_hours = hourly.index.to_series().diff().dt.total_seconds().div(3600)
    out["hourly_gap_max_h"] = float(gaps_hours.max()) if len(gaps_hours) else None
    return out


def check_lsr(periods: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """LSRの健全性: 期間カバー・整合(buy+sell=1)・欠落。"""
    out: dict[str, Any] = {}
    for period, df in sorted(periods.items()):
        sums = df["buy_ratio"] + df["sell_ratio"]
        bad_sum = int(((sums - 1.0).abs() > 1e-6).sum())
        gaps = df.index.to_series().diff()
        unit = {"15min": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(period, 3600)
        missing = int(
            ((gaps.dt.total_seconds() / unit) - 1).round().clip(lower=0).sum()
        )
        out[period] = {
            "n": int(len(df)),
            "range": [str(df.index[0]), str(df.index[-1])],
            "sum_violations": bad_sum,
            "missing_estimate": missing,
            "buy_ratio_range": [
                float(df["buy_ratio"].min()),
                float(df["buy_ratio"].max()),
            ],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="派生データ健全性チェック")
    parser.add_argument("--symbol", default=None, help="単一シンボルに限定")
    args = parser.parse_args()
    symbols = (args.symbol,) if args.symbol else SUPPORTED

    report: dict[str, Any] = {}
    for sym in symbols:
        print(f"\n===== {sym} =====")
        fr = load_funding(sym)
        fr_report = check_funding(fr)
        report.setdefault("funding", {})[sym] = fr_report
        miss = fr_report.get("missing_rate_pct")
        print(
            f"FR : n={fr_report['n']} {fr_report.get('range')} "
            f"欠落率={miss}% 異常値(|r|>1%)={fr_report.get('outliers_gt_1pct')}"
        )

        oi = load_open_interest(sym)
        oi_report = check_open_interest(oi)
        report.setdefault("open_interest", {})[sym] = oi_report
        print(f"OI : n={oi_report['n']} {oi_report.get('range')}")
        for day, info in (oi_report.get("known_cascades") or {}).items():
            print(f"     カスケード {day}: {info}")
        print(f"     最大日次下落TOP5: {oi_report.get('top5_daily_drawdowns')}")

        lsr_frames = load_lsr_periods(sym)
        lsr_report = check_lsr(lsr_frames)
        report.setdefault("lsr", {})[sym] = lsr_report
        for period, info in lsr_report.items():
            print(
                f"LSR[{period}] n={info['n']} {info['range']} "
                f"和違反={info['sum_violations']} 欠落≈{info['missing_estimate']}"
            )

    out_path = PROJECT_ROOT / "results" / "derivative_data_quality.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
