"""「sideways時のみ稼働」運用則のウォークフォワード検証.

検証内容:
  1. 日次終値からリアルタイム（先読みなし）で月次レジームラベルを計算
  2. 検証済みGA戦略（合格個体のみ）を、その学習期間の直後Nヶ月間、
     月単位でバックテストし、検出レジーム別の成績を集計
  3. 運用ルール比較:
       - R1: sideways検出月のみsideways進化戦略を稼働
       - ALL: 常時稼働 / NOT_BULL: bull月以外稼働
     各ルールの月次平均excess累計と勝率を比較

注意: 月単位で区切るため戦略は各月末で強制決済相当となり、
連続保有より取引コストが過大評価される（保守的）。

Usage:
  uv run python -m scripts.verify_regime_rule [--months 12] [--limit-genes 10] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.regime import (  # noqa: E402
    REGIME_BULL,
    REGIME_SIDEWAYS,
    RegimeDetector,
    classify_return,
    forward_return_pp,
)
from database.connection import SessionLocal  # noqa: E402
from database.repositories.ohlcv_repository import OHLCVRepository  # noqa: E402
from scripts.validate_pipeline import (  # noqa: E402
    OOSEvaluator,
    gene_hash,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "auto_strategy"
SUPPORTED = ("BTC/USDT:USDT", "ETH/USDT:USDT")


def load_daily_closes(symbol: str) -> pd.Series:
    db = SessionLocal()
    try:
        repo = OHLCVRepository(db)
        df = repo.get_ohlcv_dataframe(symbol, "1d")
    finally:
        db.close()
    if df.empty or "close" not in df.columns:
        raise RuntimeError(f"日次データなし: {symbol}")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # DBはUTC保持、比較用にnaiveへ統一
    s = pd.Series(df["close"].astype(float).values, index=idx)
    return s.sort_index()


def collect_unique_genes() -> list[dict[str, Any]]:
    """結果ファイルから合格遺伝子をユニーク化して返す。"""
    genes: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS_DIR.glob("strategy_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        bc = data.get("backtest_config") or {}
        symbol = str(bc.get("symbol") or "")
        if symbol not in SUPPORTED:
            continue
        strategies = data.get("validated_strategies") or {}
        validations = data.get("validation_results") or {}
        for gid, validation in validations.items():
            raw = strategies.get(gid)
            if raw is None or not (validation or {}).get("passed"):
                continue
            gh = gene_hash(raw)
            key = f"{gh}:{symbol}"
            if key in genes:
                continue
            genes[key] = {
                "gene_hash": gh,
                "gene": raw,
                "symbol": symbol,
                "is_start": str(bc.get("start_date")),
                "is_end": str(bc.get("end_date")),
                "backtest_config": dict(bc),
                "source": path.name,
            }
    return list(genes.values())


def month_range(start_after: str, months: int) -> list[pd.Timestamp]:
    """start_after の翌月から months 個分の月初を返す。"""
    first_is_end = pd.Timestamp(start_after)
    next_month = first_is_end.replace(day=1) + pd.offsets.MonthBegin(1)
    return [next_month + pd.offsets.MonthBegin(i) for i in range(months)]


def month_bounds(
    month_start: pd.Timestamp, last_data_ts: pd.Timestamp
) -> tuple[str, str] | None:
    """月の [開始日, 終了日]。データが月末まで無い場合は利用可能範囲に短縮。"""
    m_end = month_start + pd.offsets.MonthEnd(0)
    eff_end = min(m_end, last_data_ts.normalize())
    if eff_end < month_start + pd.Timedelta(days=20):
        return None  # 実質データが足りない月
    return month_start.strftime("%Y-%m-%d"), eff_end.strftime("%Y-%m-%d")


def cumulative_curve(monthly_excess: list[float]) -> tuple[float, float]:
    """月次excess(pp)系列の累計と最大ドローダウン(pp)を返す。"""
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in monthly_excess:
        cum += v
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return cum, max_dd


def summarize(name: str, values: list[float]) -> None:
    if not values:
        print(f"  {name:34s} n=0")
        return
    wins = sum(1 for v in values if v > 0)
    cum, dd = cumulative_curve(values)
    print(
        f"  {name:34s} n={len(values):>3d} "
        f"月次中央値 {st.median(values):+6.2f}pp / 勝率 {wins}/{len(values)}"
        f" / 累計 {cum:+8.1f}pp / 最大DD {dd:+7.1f}pp"
    )


def make_evaluator() -> tuple[OOSEvaluator, Any, Any]:
    """早期終了を無効化した評価器を返す。

    月次窓では取引数が少なく early_terminated で指標が欠ける月が多発し、
    除外がレジームと相関するとサンプルにバイアスが掛かるため、
    全月を強制評価する（ハーネスの半年窓とはこの点で異なる）。
    """
    evaluator = OOSEvaluator()
    es, serializer, ga_config = evaluator._ensure()
    try:
        import dataclasses

        ec = dataclasses.replace(
            ga_config.evaluation_config,
            early_termination_settings=dataclasses.replace(
                ga_config.evaluation_config.early_termination_settings,
                enabled=False,
            ),
        )
        ga_config = dataclasses.replace(ga_config, evaluation_config=ec)
    except Exception:  # noqa: BLE001
        pass  # 構造変更時はデフォルト挙動へフォールバック
    return evaluator, es, ga_config


def evaluate_full(
    evaluator: OOSEvaluator,
    es: Any,
    ga_config: Any,
    gene_raw: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """早期終了無効版の単一評価（OOSEvaluator.evaluate と同じ指標抽出）。"""
    try:
        _, serializer, _ = evaluator._ensure()
        gene = serializer.dict_to_strategy_gene(gene_raw)
        report = es.execute_single_report(gene, dict(cfg), ga_config)
        if not report.scenarios:
            return None
        pm = report.scenarios[0].performance_metrics or {}
        fa = (report.scenarios[0].metadata or {}).get("funding_adjustment") or {}
        ret, bh = pm.get("total_return"), pm.get("buy_hold_return")
        trades = pm.get("total_trades")
        if not (isinstance(ret, (int, float)) and isinstance(bh, (int, float))):
            return None
        return {
            "oos_excess_pp": (float(ret) - float(bh)) * 100.0,
            "oos_return_pp": float(ret) * 100.0,
            "oos_bh_pp": float(bh) * 100.0,
            "trades": int(trades or 0),
            "drag_pp": float(fa.get("strategy_drag_pct") or 0.0),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"    評価例外: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="レジーム運用則ウォークフォワード検証")
    parser.add_argument("--months", type=int, default=12, help="遺伝子ごとの評価月数")
    parser.add_argument(
        "--limit-genes", type=int, default=0, help="遺伝子数上限（0=無制限）"
    )
    parser.add_argument("--smoke", action="store_true", help="1遺伝子×2ヶ月の疎通確認")
    parser.add_argument("--debug", action="store_true", help="評価失敗の詳細を出力")
    args = parser.parse_args()
    if args.smoke:
        args.limit_genes = 1
        args.months = 2

    t0 = time.time()
    closes_by_symbol = {sym: load_daily_closes(sym) for sym in SUPPORTED}
    detector = RegimeDetector()

    # --- 検出器の基本統計（月次ラベル分布・切替回数） ---
    print("=== レジーム検出器の基本統計（ウォークフォワード） ===")
    for sym, closes in closes_by_symbol.items():
        daily_labels = detector.detect_series(closes)
        known = daily_labels.dropna()
        monthly_starts = pd.date_range(
            closes.index[0] + pd.offsets.MonthBegin(1),
            closes.index[-1],
            freq="MS",
        )
        mlabels = [
            v
            for v in detector.monthly_labels(closes, list(monthly_starts)).values()
            if v
        ]
        transitions = sum(1 for a, b in zip(known, list(known)[1:]) if a != b)
        print(
            f"{sym}: 月次ラベル n={len(mlabels)} 分布={dict(Counter(mlabels))} "
            f"/ 日次ラベル切替回数={transitions} ({len(known)}日中)"
        )

    # --- 遺伝子収集 ---
    genes = collect_unique_genes()
    print(f"\n合格ユニーク遺伝子: {len(genes)} 件")
    if args.limit_genes > 0:
        genes = genes[: args.limit_genes]
        print(f"  --limit-genes により {len(genes)} 件に制限")

    evaluator, es, ga_config = make_evaluator()
    records: list[dict[str, Any]] = []
    errors = 0
    for i, g in enumerate(genes):
        closes = closes_by_symbol[g["symbol"]]
        is_ret = forward_return_pp(
            closes,
            pd.Timestamp(g["is_start"]),
            pd.Timestamp(g["is_end"]),
        )
        g["is_regime"] = classify_return(is_ret) if is_ret is not None else None
        last_ts = closes.index[-1]
        for ms in month_range(g["is_end"], args.months):
            bounds = month_bounds(ms, last_ts)
            if bounds is None:
                continue
            label = detector.label_at(closes, ms - pd.Timedelta(days=1))
            if label is None:
                continue
            outcome = evaluate_full(
                evaluator,
                es,
                ga_config,
                g["gene"],
                {
                    **g["backtest_config"],
                    "start_date": bounds[0],
                    "end_date": bounds[1],
                },
            )
            if outcome is None:
                errors += 1
                if args.debug:
                    print(
                        f"    [debug] {g['gene_hash']} {bounds} "
                        f"評価不可（scenarios空 or 指標欠落）"
                    )
                continue
            records.append(
                {
                    "gene_hash": g["gene_hash"],
                    "symbol": g["symbol"],
                    "month": ms.strftime("%Y-%m"),
                    "detected_regime": label,
                    "is_regime": g["is_regime"],
                    **outcome,
                }
            )
        if (i + 1) % 5 == 0 or i + 1 == len(genes):
            print(
                f"  ... {i + 1}/{len(genes)} 遺伝子処理済み "
                f"(観測 {len(records)} 件, 経過 {time.time() - t0:.0f}s)"
            )

    if not records:
        print("観測なし — データ/期間設定を確認してください")
        return

    # --- 集計 ---
    print(f"\n=== 検出レジーム別 成績（全遺伝子・月次観測 n={len(records)}） ===")
    for reg in (REGIME_BULL, REGIME_SIDEWAYS, "bear"):
        sub = [r["oos_excess_pp"] for r in records if r["detected_regime"] == reg]
        summarize(f"検出={reg}", sub)

    side_genes = [g for g in genes if g.get("is_regime") == REGIME_SIDEWAYS]
    print(f"\n=== 運用ルール比較（sideways進化遺伝子 {len(side_genes)} 本に限定） ===")
    side_hashes = {g["gene_hash"] for g in side_genes}
    side_records = [r for r in records if r["gene_hash"] in side_hashes]

    rules: dict[str, list[float]] = {
        "ALL(常時)": [],
        "NOT_BULL(bull以外)": [],
        "R1(sidewaysのみ)": [],
    }
    for r in side_records:
        ex = float(r["oos_excess_pp"])
        rules["ALL(常時)"].append(ex)
        if r["detected_regime"] != REGIME_BULL:
            rules["NOT_BULL(bull以外)"].append(ex)
        if r["detected_regime"] == REGIME_SIDEWAYS:
            rules["R1(sidewaysのみ)"].append(ex)
    for name, vals in rules.items():
        summarize(name, vals)

    print("\n--- 参考: IS進化レジーム × 検出レジーム（観測数） ---")
    cross: Counter = Counter()
    cells: dict[tuple[str, str], list[float]] = {}
    for r in records:
        ir, dr = str(r.get("is_regime")), r["detected_regime"]
        cross[(ir, dr)] += 1
        cells.setdefault((ir, dr), []).append(float(r["oos_excess_pp"]))
    for (ir, dr), n in sorted(cross.items()):
        vals = cells[(ir, dr)]
        wins = sum(1 for v in vals if v > 0)
        print(
            f"  IS={ir:9s} → 検出={dr:9s} : n={n:>4d} "
            f"中央値 {st.median(vals):+7.2f}pp / 勝ち {wins}/{n}"
        )

    out_path = PROJECT_ROOT / "results" / "regime_rule_records.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n観測を保存: {out_path}")

    print(
        f"\n完了: 観測 {len(records)} 件 / 評価エラー {errors} 件 / 所要 {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
