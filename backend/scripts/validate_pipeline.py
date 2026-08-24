"""パイプライン検証ハーネス（恒久版）.

マルチ期間のGA実行 → 合格候補のOOS評価 → 指標比較を一コマンドで実行し、
候補単位の記録を results/validation_history.jsonl に蓄積する。

使い方:
    # 標準セット（ベア/ブル/横ばい）をGA実行込みで検証
    uv run python -m scripts.validate_pipeline

    # 既存の最新結果ファイルを再利用してOOS評価とメタ分析のみ実行
    uv run python -m scripts.validate_pipeline --skip-ga

    # 期間指定
    uv run python -m scripts.validate_pipeline --periods 2022H1,2024H1

蓄積された履歴に対して、選抜キー仮説（レジーム最小マージンのOOS予測力）
を n の伸びとともに追跡できる。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics as st
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HEX8 = re.compile(r"[0-9a-f]{8}")
UID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
PENALTY_THRESHOLD = -1e8

DEFAULT_PERIODS = ("2022H1", "2023H1", "2024H1")
PERIOD_REGISTRY: dict[str, tuple[str, str]] = {
    "2022H1": ("2022-01-01", "2022-06-30"),
    "2022H2": ("2022-07-01", "2022-12-31"),
    "2023H1": ("2023-01-01", "2023-06-30"),
    "2023H2": ("2023-07-01", "2023-12-31"),
    "2024H1": ("2024-01-01", "2024-06-30"),
    "2024H2": ("2024-07-01", "2024-12-31"),
    "2025H1": ("2025-01-01", "2025-06-30"),
    "2025H2": ("2025-07-01", "2025-12-31"),
    "2026H1": ("2026-01-01", "2026-06-30"),
}
SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "4h"
BASE_CONFIG = {
    "symbol": SYMBOL,
    "timeframe": TIMEFRAME,
    "initial_capital": 100000.0,
    "commission_rate": 0.0004,
    "slippage": 0.0001,
}


# ---------------------------------------------------------------------------
# 純粋ヘルパー（ユニットテスト対象）
# ---------------------------------------------------------------------------
def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson 相関係数（n<3 または分散ゼロで NaN）。"""
    n = len(xs)
    if n < 3 or len(xs) != len(ys):
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def rank(xs: list[float]) -> list[float]:
    """平均ランク（タイ処理済み）。Spearman 用。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank(xs), rank(ys))


def norm_gene(gene: Any) -> str:
    """遺伝子辞書を ID/タイムスタンプ非依存の正規化文字列へ変換する。"""

    def _walk(obj: Any) -> str:
        if isinstance(obj, dict):
            inner = {
                k: _walk(v)
                for k, v in obj.items()
                if k not in ("generated_at", "assembled_at", "id")
            }
            return "{" + ",".join(f"{k}:{v}" for k, v in sorted(inner.items())) + "}"
        if isinstance(obj, list):
            return "[" + ",".join(_walk(x) for x in obj) + "]"
        if isinstance(obj, str):
            return '"' + HEX8.sub("H", UID.sub("U", obj)) + '"'
        return json.dumps(obj, sort_keys=True)

    return _walk(gene)


def gene_hash(raw_gene: Any) -> str:
    """遺伝子をID/タイムスタンプ非依存の12桁ハッシュへ変換する。"""
    return hashlib.sha256(norm_gene(raw_gene).encode("utf-8")).hexdigest()[:12]


def record_key(row: dict[str, Any]) -> tuple:
    """履歴重複判定キー。gene_hash優先、旧レコードは指標タプルで代替する。

    評価シンボルもキーに含む（同一遺伝子でもBTC-OOSとETH-OOSは別観測）。
    """
    gh = row.get("gene_hash")
    sym = str(row.get("eval_symbol") or SYMBOL)
    if gh:
        return ("h", str(gh), sym)
    return (
        "m",
        row.get("period"),
        row.get("fitness"),
        row.get("regime_min_margin_pp"),
        row.get("pert_survival"),
        row.get("is_excess_pp"),
        sym,
    )


def select_unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """履歴行からラン間重複（同一遺伝子の再記録）を除去する。"""
    seen: set[tuple] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = record_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def regime_min_margin_pp(validation: dict[str, Any]) -> float | None:
    """レジーム一貫性スライスの最小マージン（pp）。未記録は None。"""
    rc = validation.get("regime_consistency") or {}
    slices = rc.get("slices") or []
    margins = [
        float(s["excess"])
        for s in slices
        if isinstance(s, dict) and isinstance(s.get("excess"), (int, float))
    ]
    return min(margins) * 100.0 if margins else None


def extract_candidate_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    """検証結果からユニーク候補ごとの指標レコードを抽出する。"""
    strategies = result.get("validated_strategies") or {}
    validations = result.get("validation_results") or {}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for gid, validation in validations.items():
        raw = strategies.get(gid)
        if raw is None:
            continue
        key = norm_gene(raw)
        if key in seen:
            continue
        seen.add(key)
        fit = validation.get("primary_fitness")
        rs = validation.get("report_summary") or {}
        scenarios = rs.get("scenarios") or []
        pm = (
            scenarios[0].get("performance_metrics", {})
            if scenarios and isinstance(scenarios[0], dict)
            else {}
        )
        ret, bh = pm.get("total_return"), pm.get("buy_hold_return")
        is_ex = (
            (float(ret) - float(bh)) * 100.0
            if isinstance(ret, (int, float)) and isinstance(bh, (int, float))
            else None
        )
        pp = validation.get("param_perturbation") or {}
        records.append(
            {
                "gid": str(gid)[:8],
                "gene_hash": gene_hash(raw),
                "passed": bool(validation.get("passed", False)),
                "fitness": (
                    float(fit)
                    if isinstance(fit, (int, float))
                    and math.isfinite(float(fit))
                    and float(fit) > PENALTY_THRESHOLD
                    else None
                ),
                "regime_min_margin_pp": regime_min_margin_pp(validation),
                "pert_survival": (
                    float(pp["survival_rate"])
                    if isinstance(pp.get("survival_rate"), (int, float))
                    else None
                ),
                "is_excess_pp": is_ex,
                "reasons": list(validation.get("reasons") or [])[:3],
            }
        )
    return records


def build_oos_config(
    backtest_config: dict[str, Any],
    symbol: str | None = None,
) -> dict[str, Any] | None:
    """IS期間と同長の直後窓を持つ OOS 設定を返す。

    symbol を指定すると別シンボル上での OOS 評価（クロスシンボル検証）
    設定を構築する。未指定時は元設定のシンボル（無ければ BTC）を使う。
    """
    s, e = backtest_config.get("start_date"), backtest_config.get("end_date")
    if not s or not e:
        return None
    try:
        sd, ed = datetime.fromisoformat(str(s)), datetime.fromisoformat(str(e))
    except ValueError:
        return None
    dur = ed - sd
    if dur <= timedelta(0):
        return None
    return {
        **BASE_CONFIG,
        "symbol": symbol or str(backtest_config.get("symbol") or SYMBOL),
        "start_date": (ed + timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date": (ed + dur + timedelta(days=1)).strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# 実行系
# ---------------------------------------------------------------------------
def run_ga_for_period(
    start: str,
    end: str,
    log_path: Path | None,
    symbol: str = SYMBOL,
) -> Path | None:
    """scripts.run_auto_strategy をサブプロセス実行し結果JSONパスを返す。"""
    before = set((PROJECT_ROOT / "results/auto_strategy").glob("strategy_*.json"))
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_auto_strategy",
        "--symbol",
        symbol,
        "--start-date",
        start,
        "--end-date",
        end,
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if log_path is not None:
        log_path.write_text(
            (proc.stdout or "") + "\n===STDERR===\n" + (proc.stderr or ""),
            encoding="utf-8",
        )
    if proc.returncode != 0:
        print(f"  GA実行失敗 (rc={proc.returncode}): {log_path}")
        return None
    after = set((PROJECT_ROOT / "results/auto_strategy").glob("strategy_*.json"))
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime)
    return new_files[-1] if new_files else None


def find_latest_result_for_period(
    start: str,
    end: str,
    symbol: str = SYMBOL,
) -> Path | None:
    """既存ファイルから該当期間・シンボルの最新結果を探す（--skip-ga 用）。"""
    best: tuple[float, Path] | None = None
    for f in (PROJECT_ROOT / "results/auto_strategy").glob("strategy_*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        bc = d.get("backtest_config") or {}
        if (
            bc.get("start_date") == start
            and bc.get("end_date") == end
            and str(bc.get("symbol")) == symbol
        ):
            mtime = f.stat().st_mtime
            if best is None or mtime > best[0]:
                best = (mtime, f)
    return best[1] if best else None


class OOSEvaluator:
    """合格候補の OOS 評価を担う（重い依存は遅延生成）。"""

    def __init__(self) -> None:
        self._es: Any = None
        self._serializer: Any = None
        self._ga_config: Any = None

    def _ensure(self) -> tuple[Any, Any, Any]:
        if self._es is None:
            from app.services.auto_strategy.config.ga_config import GAConfig
            from app.services.auto_strategy.core.evaluation.evaluation_strategies import (
                EvaluationStrategy,
            )
            from app.services.auto_strategy.core.evaluation.individual_evaluator import (
                IndividualEvaluator,
            )
            from app.services.auto_strategy.serializers.serialization import (
                GeneSerializer,
            )
            from app.services.backtest.services.backtest_service import BacktestService

            service = BacktestService()
            evaluator = IndividualEvaluator(service)
            self._es = EvaluationStrategy(evaluator, max_workers=2)
            self._serializer = GeneSerializer()
            self._ga_config = GAConfig()
        return self._es, self._serializer, self._ga_config

    def evaluate(
        self, gene_raw: dict[str, Any], oos_cfg: dict[str, Any]
    ) -> dict | None:
        es, serializer, ga_config = self._ensure()
        try:
            gene = serializer.dict_to_strategy_gene(gene_raw)
            report = es.execute_single_report(gene, dict(oos_cfg), ga_config)
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
            print(f"    OOS評価エラー: {exc}")
            return None


def append_history(path: Path, records: list[dict[str, Any]]) -> int:
    """履歴へ追記する。gene_hash等で重複する既存・バッチ内レコードはスキップ。"""
    existing = load_history(path)
    seen = {record_key(row) for row in existing}
    fresh: list[dict[str, Any]] = []
    for rec in records:
        key = record_key(rec)
        if key in seen:
            continue
        seen.add(key)
        fresh.append(rec)
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as fh:
        for rec in fresh:
            fh.write(json.dumps({"ts": ts, **rec}, ensure_ascii=False) + "\n")
    return len(fresh)


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def meta_report(rows: list[dict[str, Any]]) -> None:
    """履歴全体での「選抜キー候補 ↔ OOS」予測力トラッキング（重複除去済み）。

    同シンボルOOSのみで相関判定し、クロスシンボル行は別サマリに出力する。
    """
    rows = select_unique_rows(rows)
    cross_rows = [r for r in rows if r.get("cross_symbol")]
    same_symbol = [r for r in rows if not r.get("cross_symbol")]
    usable = [
        r for r in same_symbol if isinstance(r.get("oos_excess_pp"), (int, float))
    ]
    print("\n===== メタ分析トラッカー =====")
    if len(usable) < 10:
        print(f"ユニークn={len(usable)} / 最低10件で判定開始")
        _cross_summary(cross_rows)
        return
    metrics = ("fitness", "regime_min_margin_pp", "is_excess_pp")
    print(f"{'指標':22s} {'n':>4s} {'Pearson':>8s} {'Spearman':>8s}")
    for m in metrics:
        pairs = [
            (float(r[m]), float(r["oos_excess_pp"]))
            for r in usable
            if isinstance(r.get(m), (int, float))
        ]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        if len(pairs) >= 3:
            print(
                f"{m:22s} {len(pairs):>4d} "
                f"{pearson(xs, ys):>+8.2f} {spearman(xs, ys):>+8.2f}"
            )
        else:
            print(f"{m:22s} {len(pairs):>4d}       ---        ---")
    _cross_summary(cross_rows)


def _cross_summary(cross_rows: list[dict[str, Any]]) -> None:
    """クロスシンボルOOS行のサマリを出力する（存在時のみ）。"""
    usable = [r for r in cross_rows if isinstance(r.get("oos_excess_pp"), (int, float))]
    if not usable:
        return
    exs = [float(r["oos_excess_pp"]) for r in usable]
    beats = sum(1 for v in exs if v > 0)
    symbols = sorted({str(r.get("eval_symbol")) for r in usable})
    print(
        f"クロスシンボルOOS n={len(usable)} ({','.join(symbols)}): "
        f"中央値 {st.median(exs):+.2f}pp / 市場比勝ち {beats}/{len(usable)}"
    )


def process_period(
    label: str,
    evaluator: OOSEvaluator,
    history_records: list[dict[str, Any]],
    skip_ga: bool,
    eval_symbol: str | None = None,
    ga_symbol: str = SYMBOL,
) -> bool:
    start, end = PERIOD_REGISTRY[label]
    if skip_ga:
        result_path = find_latest_result_for_period(start, end, symbol=ga_symbol)
        if result_path is None:
            print(
                f"\n[{label}] 既存結果なし（symbol={ga_symbol} / --skip-ga解除で実行）"
            )
            return False
    else:
        print(f"\n[{label}] GA実行中 ({start} ~ {end}, symbol={ga_symbol}) ...")
        t0 = time.time()
        result_path = run_ga_for_period(
            start,
            end,
            PROJECT_ROOT / "logs" / f"validate_pipeline_{label}.log",
            symbol=ga_symbol,
        )
        if result_path is None:
            print(f"[{label}] 結果JSONが生成されず")
            return False
        print(f"  実行時間 {time.time() - t0:.0f}s -> {result_path.name}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    records = extract_candidate_records(result)
    passed = [r for r in records if r["passed"]]
    print(
        f"  検証 {len(records)} 体 / 合格 {len(passed)} 体"
        + ("" if not records else "")
    )

    reason_hist: dict[str, int] = {}
    for r in records:
        if not r["passed"] and r["reasons"]:
            head = r["reasons"][0][:24]
            reason_hist[head] = reason_hist.get(head, 0) + 1
    for reason, count in sorted(reason_hist.items(), key=lambda kv: -kv[1])[:3]:
        print(f"    不合格理由TOP: {reason}... x{count}")

    oos_cfg = build_oos_config(result.get("backtest_config") or {}, symbol=eval_symbol)
    if oos_cfg is None:
        print("  OOS設定を構築できず（スキップ）")
        return False
    symbol_note = f" on {eval_symbol}" if eval_symbol else ""
    print(
        f"  OOS{symbol_note}: {oos_cfg['start_date']} ~ {oos_cfg['end_date']} "
        f"で合格 {len(passed)} 体を評価"
    )

    oos_rows: list[dict[str, Any]] = []
    for r in passed:
        raw = next(
            (
                s
                for gid, s in (result.get("validated_strategies") or {}).items()
                if str(gid)[:8] == r["gid"]
            ),
            None,
        )
        if raw is None:
            continue
        outcome = evaluator.evaluate(raw, oos_cfg)
        if outcome is None:
            continue
        row = {"period": label, "ga_symbol": ga_symbol, **r, **outcome}
        if eval_symbol:
            row["eval_symbol"] = eval_symbol
            row["cross_symbol"] = eval_symbol != SYMBOL
        oos_rows.append(row)
        history_records.append(row)

    if not oos_rows:
        print("  OOS評価できた合格者なし")
        return False

    exs = [r["oos_excess_pp"] for r in oos_rows]
    beats = sum(1 for v in exs if v > 0)
    absolutes = sum(1 for r in oos_rows if r["oos_return_pp"] > 0)
    print(
        f"  OOS市場比 中央値 {st.median(exs):+.2f}pp / B&H勝ち {beats}/{len(oos_rows)}"
        f" / 絶対黒字 {absolutes}/{len(oos_rows)}"
    )
    for r in oos_rows:
        print(
            f"    [{r['gid']}] margin={r['regime_min_margin_pp']} "
            f"fit={r['fitness']} IS={r['is_excess_pp']:+.1f}pp "
            f"-> OOS={r['oos_excess_pp']:+.1f}pp (ret{r['oos_return_pp']:+.1f}% tr{r['trades']})"
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="パイプライン検証ハーネス")
    parser.add_argument(
        "--periods",
        default=",".join(DEFAULT_PERIODS),
        help="カンマ区切りの期間ラベル",
    )
    parser.add_argument(
        "--skip-ga",
        action="store_true",
        help="GAを実行せず既存の最新結果を使用",
    )
    parser.add_argument(
        "--history",
        default="results/validation_history.jsonl",
        help="候補記録の蓄積先(JSONL)",
    )
    parser.add_argument(
        "--eval-symbol",
        default=None,
        help=(
            "OOS評価を別シンボル上で実行（クロスシンボル検証）。"
            "例: ETH/USDT:USDT。履歴には cross_symbol フラグが付く"
        ),
    )
    parser.add_argument(
        "--symbol",
        default=SYMBOL,
        help=f"GA実行・結果検索の対象シンボル（既定: {SYMBOL}）",
    )
    args = parser.parse_args()

    labels = [p.strip() for p in args.periods.split(",") if p.strip()]
    unknown = [lab for lab in labels if lab not in PERIOD_REGISTRY]
    if unknown:
        parser.error(f"未知の期間: {unknown}")

    history_path = PROJECT_ROOT / args.history
    evaluator = OOSEvaluator()
    new_records: list[dict[str, Any]] = []
    ok_count = 0
    started = time.time()
    for label in labels:
        try:
            if process_period(
                label,
                evaluator,
                new_records,
                args.skip_ga,
                eval_symbol=args.eval_symbol,
                ga_symbol=args.symbol,
            ):
                ok_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}] 処理エラー: {exc}")

    if new_records:
        added = append_history(history_path, new_records)
        skipped = len(new_records) - added
        print(
            f"\n履歴追記: {added}件（遺伝子重複スキップ {skipped}件） -> {history_path}"
        )

    print(f"\n完了: {ok_count}/{len(labels)} 期間処理 ({time.time() - started:.0f}s)")
    meta_report(load_history(history_path))


if __name__ == "__main__":
    main()
