"""
エッジ検証スクリプト (IS→OOS 順位相関)

GA を学習窓(IS)で進化させ、最終集団を進化に一切使用していない
評価窓(OOS)で再評価する。「IS適応度の順位がOOSでも保つか」を
順位相関で測り、ランダム遺伝子群・バイトアンドホールドと比較して
探索が本物のシグナルを掴んでいるかを判定する。

使用例:
    uv run python scripts/verify_edge_oos.py --population 50 --generations 30
    uv run python scripts/verify_edge_oos.py --smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# tqdm のプログレスバーは1バーあたり数百回の書き込みが発生し、
# 長期窓の計測では実行時間が数倍に伸びるため無効化する
os.environ.setdefault("TQDM_DISABLE", "1")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.auto_strategy.config.ga_config import (  # noqa: E402
    EarlyTerminationSettings,
    EvaluationConfig,
    FitnessSharingConfig,
    GAConfig,
    SurvivalSelectionConfig,
)
from app.services.auto_strategy.core import GeneticAlgorithmEngine  # noqa: E402
from app.services.auto_strategy.core.engine.fitness_utils import (  # noqa: E402
    extract_individual_primary_fitness,
)
from app.services.auto_strategy.core.evaluation.evaluation_report import (  # noqa: E402
    _is_penalty_value,
)
from app.services.auto_strategy.core.evaluation.individual_evaluator import (  # noqa: E402
    IndividualEvaluator,
)
from app.services.auto_strategy.generators.random_gene_generator import (  # noqa: E402
    RandomGeneGenerator,
)
from app.services.auto_strategy.utils.strategy_dedup import (  # noqa: E402
    strategy_structure_signature,
)
from app.services.backtest.services.backtest_service import (  # noqa: E402
    BacktestService,
)
from database.connection import SessionLocal  # noqa: E402
from database.models import OHLCVData  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# トレード毎・バー毎のログは長期窓ではログ洪水になるため抑制する
for _noisy in (
    "app.services.auto_strategy.strategies",
    "app.services.backtest",
    "app.services.auto_strategy.core.evaluation",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "4h"


def build_config(args: argparse.Namespace) -> GAConfig:
    """計測と同一条件 (sharing+RTR) の GAConfig を構築する。"""
    preset = getattr(args, "objectives", "weighted_score")
    kwargs: dict[str, Any] = {
        "population_size": 4 if args.smoke else args.population,
        "generations": 1 if args.smoke else args.generations,
        "crossover_rate": 0.8,
        "mutation_rate": 0.2,
        "elite_size": 2,
        "max_indicators": 10,
        "max_conditions": 3,
        "non_price_indicator_probability": 0.3,
        "use_seed_strategies": args.seed_rate > 0.0,
        "seed_injection_rate": args.seed_rate if args.seed_rate > 0.0 else 0.0,
        "fitness_sharing": FitnessSharingConfig(),
        "survival_selection_config": SurvivalSelectionConfig(
            enable_restricted_tournament=True,
            restricted_tournament_crowding_factor=args.rtr_cf,
        ),
        "evaluation_config": EvaluationConfig(enable_parallel=not args.smoke),
        "fallback_start_date": args.is_start,
        "fallback_end_date": args.is_end,
    }
    if preset == "excess_return":
        kwargs["objectives"] = ["excess_return"]
        kwargs["objective_weights"] = [1.0]
        kwargs["objective_preset"] = "excess_return"
    if getattr(args, "seed", None) is not None:
        kwargs["random_state"] = int(args.seed)
    if getattr(args, "wfa_ga", False):
        # ValidationConfig の WFA 設定を流用 (ユーザ選択: ValidationConfig流用)
        kwargs["evaluation_config"] = EvaluationConfig(
            enable_parallel=not args.smoke, enable_walk_forward_for_ga=True
        )
    if getattr(args, "no_early_termination", False):
        # WFA-GA のフォールド評価で早期終了 (DD/期待値/ペース打ち切り) を無効化する
        kwargs["evaluation_config"] = EvaluationConfig(
            enable_parallel=not args.smoke,
            enable_walk_forward_for_ga=getattr(args, "wfa_ga", False),
            early_termination_settings=EarlyTerminationSettings(enabled=False),
        )
    return GAConfig(**kwargs)


def build_backtest_config(start: str, end: str) -> dict[str, Any]:
    return {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "start_date": start,
        "end_date": end,
        "initial_capital": 100000.0,
        "commission_rate": 0.0004,
        "slippage": 0.0001,
    }


def evaluate_genes(
    evaluator: IndividualEvaluator,
    genes: list[Any],
    config: GAConfig,
    label: str,
    workers: int,
) -> list[tuple[float, ...] | None]:
    """遺伝子リストを並列評価し、失敗は None として結果を返す。"""
    results: list[tuple[float, ...] | None] = [None] * len(genes)

    def work(index: int) -> tuple[int, tuple[float, ...] | None]:
        try:
            return index, evaluator.evaluate(genes[index], config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 評価失敗 #%d: %s", label, index, exc)
            return index, None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        for index, fitness in pool.map(work, range(len(genes))):
            results[index] = fitness
    logger.info("[%s] %d 個体を %.0fs で評価", label, len(genes), time.time() - t0)
    return results


def valid_score(value: float | None, penalty_values: frozenset[float]) -> float | None:
    """非有限値・±1e9 ペナルティ・制約違反ペナルティ定数を除外して返す。"""
    if value is None:
        return None
    if value != value or abs(value) == float("inf"):
        return None
    if _is_penalty_value(value):
        return None
    # zero_trades / constraint_violation ペナルティは有限の定数値
    # (デフォルト 0.1 / 0.0) で返るため定数一致で除外する
    if any(value == p for p in penalty_values):
        return None
    return value


def primary(
    fitness: tuple[float, ...] | None, penalty_values: frozenset[float]
) -> float | None:
    """fitness タプルから主目的の有効値を取り出す(ペナルティ値は除外)。"""
    if not fitness:
        return None
    try:
        value = float(fitness[0])
    except (TypeError, ValueError):
        return None
    return valid_score(value, penalty_values)


def buy_and_hold_return(start: str, end: str) -> float | None:
    """評価窓のバイトアンドホールド価格リターンをDBから計算する。"""
    start_dt = datetime.fromisoformat(f"{start}T00:00:00+00:00")
    end_dt = datetime.fromisoformat(f"{end}T00:00:00+00:00")
    with SessionLocal() as db:
        base = (
            db.query(OHLCVData)
            .filter(
                OHLCVData.symbol == SYMBOL,
                OHLCVData.timeframe == TIMEFRAME,
                OHLCVData.timestamp >= start_dt,
                OHLCVData.timestamp <= end_dt,
            )
            .order_by(OHLCVData.timestamp.asc())
        )
        first = base.first()
        last = (
            db.query(OHLCVData)
            .filter(
                OHLCVData.symbol == SYMBOL,
                OHLCVData.timeframe == TIMEFRAME,
                OHLCVData.timestamp >= start_dt,
                OHLCVData.timestamp <= end_dt,
            )
            .order_by(OHLCVData.timestamp.desc())
            .first()
        )
    if first is None or last is None or first.close <= 0:
        return None
    return last.close / first.close - 1.0


def rank_correlation(
    xs: list[float], ys: list[float], *, n_boot: int = 2000, seed: int = 20260817
) -> dict[str, Any]:
    """Spearman/Kendall 順位相関とブートストラップ信頼区間を返す。"""
    out: dict[str, Any] = {
        "n": len(xs),
        "spearman_rho": None,
        "spearman_p": None,
        "kendall_tau": None,
        "kendall_p": None,
        "bootstrap_ci95": None,
    }
    if len(xs) < 3:
        return out

    rho, p = stats.spearmanr(xs, ys)
    tau, tau_p = stats.kendalltau(xs, ys)
    out["spearman_rho"] = None if rho != rho else float(rho)
    out["spearman_p"] = None if p != p else float(p)
    out["kendall_tau"] = None if tau != tau else float(tau)
    out["kendall_p"] = None if tau_p != tau_p else float(tau_p)

    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(xs)) for _ in range(len(xs))]
        sx = [xs[i] for i in idx]
        sy = [ys[i] for i in idx]
        if len(set(sx)) < 2 or len(set(sy)) < 2:
            continue
        b_rho, _ = stats.spearmanr(sx, sy)
        if b_rho == b_rho:
            boot.append(float(b_rho))
    if boot:
        boot.sort()
        out["bootstrap_ci95"] = [
            float(boot[int(0.025 * len(boot))]),
            float(boot[int(0.975 * len(boot)) - 1]),
        ]
    return out


def summarize(values: list[float | None]) -> dict[str, Any]:
    finite = [v for v in values if v is not None]
    if not finite:
        return {"n": len(values), "n_finite": 0}
    finite.sort()
    n = len(finite)
    return {
        "n": len(values),
        "n_finite": n,
        "median": finite[n // 2]
        if n % 2
        else (finite[n // 2 - 1] + finite[n // 2]) / 2,
        "mean": sum(finite) / n,
        "min": finite[0],
        "max": finite[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="IS→OOS エッジ検証")
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed-rate", type=float, default=0.3)
    parser.add_argument("--rtr-cf", type=int, default=8)
    parser.add_argument("--is-start", type=str, default="2024-01-01")
    parser.add_argument("--is-end", type=str, default="2024-06-30")
    parser.add_argument("--oos-start", type=str, default="2024-07-01")
    parser.add_argument("--oos-end", type=str, default="2025-06-30")
    parser.add_argument("--n-random", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=str, default="edge_oos_result.json")
    parser.add_argument("--ignore-constraints", action="store_true")
    parser.add_argument(
        "--objectives",
        type=str,
        default="weighted_score",
        choices=["weighted_score", "excess_return"],
        help="GA目的関数: weighted_score(デフォルト) または excess_return(B&H超過を直接最適化、長期窓で識別力維持)",
    )
    parser.add_argument(
        "--wfa-ga",
        action="store_true",
        help="IS窓内で WFA (fold=ValidationConfig) を使って選抜する (enable_walk_forward_for_ga)",
    )
    parser.add_argument(
        "--no-early-termination",
        action="store_true",
        help="GA進化のフォールド評価で早期終了 (DD/期待値/trade_pace打ち切り) を無効化する",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="GA進化とランダムヌルの乱数シード (None で非固定、指定で再現可能)",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.is_start, args.is_end = "2024-01-01", "2024-02-01"
        args.oos_start, args.oos_end = "2024-02-01", "2024-03-01"
        args.n_random = 4
        args.workers = 2
        args.out = "edge_oos_smoke.json"

    config = build_config(args)
    is_bt = build_backtest_config(args.is_start, args.is_end)
    oos_bt = build_backtest_config(args.oos_start, args.oos_end)

    # --- Phase 1: 学習窓で GA 進化 (IS fitness は進化中の値をそのまま使う) ---
    logger.info(
        "Phase 1: GA進化 IS=%s..%s (pop=%d gen=%d rtr_cf=%d)",
        args.is_start,
        args.is_end,
        config.population_size,
        config.generations,
        args.rtr_cf,
    )
    engine = GeneticAlgorithmEngine(
        backtest_service=BacktestService(),
        gene_generator=RandomGeneGenerator(config=config),
    )
    t0 = time.time()
    result = engine.run_evolution(config=config, backtest_config=is_bt)
    ga_elapsed = time.time() - t0

    population = list(result.get("all_strategies", []))
    is_fitness = [extract_individual_primary_fitness(ind) for ind in population]
    penalty_values = frozenset(
        {
            float(getattr(config, "zero_trades_penalty", 0.1)),
            float(getattr(config, "constraint_violation_penalty", 0.0)),
        }
    )
    is_primary = [valid_score(v, penalty_values) for v in is_fitness]
    unique_signatures = len({strategy_structure_signature(ind) for ind in population})
    logger.info(
        "GA完了: %.0fs best_fitness=%s 最終集団=%d ユニーク構造=%d",
        ga_elapsed,
        result.get("best_fitness"),
        len(population),
        unique_signatures,
    )

    # --- Phase 2: 未使用窓で最終集団とランダム遺伝子を再評価 ---
    logger.info("Phase 2: OOS評価 %s..%s", args.oos_start, args.oos_end)
    oos_config = copy.deepcopy(config)
    # --ignore-constraints: OOS計測で制約を無効化し生リターンで相関を測る
    # (5.1案B: max_dd<20%で12ヶ月窓が全滅する測定器修復モード)
    if args.ignore_constraints:
        logger.info(
            "OOS計測: --ignore-constraints 有効 (制約をバイパスし生リターンで順位相関を計測)"
        )
        oos_config.fitness_constraints = {}

    oos_config.evaluation_config = EvaluationConfig(
        enable_parallel=False,
        # OOS 測定では窓を分割せず全期間を 1 シナリオで評価する
        oos_split_ratio=0.0,
        # 早期終了 (DD/期待値/ペース打ち切り) は進化の高速化機構であり
        # 計測窓では結果を部分窓の数字に歪めるため無効化する
        early_termination_settings=EarlyTerminationSettings(enabled=False),
    )

    evaluator = IndividualEvaluator(BacktestService(), max_cache_size=1024)
    evaluator.set_backtest_config(oos_bt)

    oos_fitness = evaluate_genes(
        evaluator, population, oos_config, "final_pop", args.workers
    )
    oos_primary = [primary(f, penalty_values) for f in oos_fitness]

    if args.seed is not None:
        # 進化とランダムヌルで同じシード列にならないようオフセットを付ける
        random.seed(args.seed + 1)
        np.random.seed(args.seed + 1)
    generator = RandomGeneGenerator(config=config)
    random_genes = [generator.generate_random_gene() for _ in range(args.n_random)]
    rand_fitness = evaluate_genes(
        evaluator, random_genes, oos_config, "random_null", args.workers
    )
    rand_primary = [primary(f, penalty_values) for f in rand_fitness]

    # 評価レポートから OOS 側の生メトリクス (total_return 等) を取得
    def metric_of(gene: Any, key: str) -> float | None:
        report = evaluator.get_cached_evaluation_report(gene)
        if report is None or not report.scenarios:
            return None
        raw = report.scenarios[0].performance_metrics.get(key)
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return value if value == value and abs(value) != float("inf") else None

    pop_returns = [metric_of(gene, "total_return") for gene in population]
    rand_returns = [metric_of(gene, "total_return") for gene in random_genes]
    pop_trades = [metric_of(gene, "total_trades") for gene in population]
    rand_trades = [metric_of(gene, "total_trades") for gene in random_genes]
    pop_max_dd = [metric_of(gene, "max_drawdown") for gene in population]
    bh_return = buy_and_hold_return(args.oos_start, args.oos_end)

    # --- Phase 3: 集計 ---
    pairs = [
        (i, o)
        for i, o in zip(is_primary, oos_primary, strict=True)
        if i is not None and o is not None
    ]
    corr = rank_correlation([i for i, _ in pairs], [o for _, o in pairs])

    evolved_finite = [v for v in oos_primary if v is not None]
    random_finite = [v for v in rand_primary if v is not None]
    mwu_p = None
    if evolved_finite and random_finite:
        mwu_p = float(
            stats.mannwhitneyu(
                evolved_finite, random_finite, alternative="greater"
            ).pvalue
        )

    # IS 上位個体 (構造クローンは重複排除) が OOS でも上位半分に残る率
    ranked = sorted(
        (i for i, v in enumerate(is_primary) if v is not None),
        key=lambda i: is_primary[i],  # type: ignore[index, return-value]
        reverse=True,
    )
    seen_signatures: set[tuple] = set()
    unique_ranked: list[int] = []
    for i in ranked:
        signature = strategy_structure_signature(population[i])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_ranked.append(i)
    oos_finite_sorted = sorted(evolved_finite)
    top_k = min(10, len(unique_ranked))

    def oos_percentile(value: float) -> float:
        if not oos_finite_sorted:
            return float("nan")
        below = sum(1 for v in oos_finite_sorted if v <= value)
        return 100.0 * below / len(oos_finite_sorted)

    top_rows = []
    for i in unique_ranked[:top_k]:
        o = oos_primary[i]
        if o is not None:
            top_rows.append(
                {
                    "is_fitness": is_primary[i],
                    "oos_fitness": o,
                    "oos_percentile": oos_percentile(o),
                    "oos_total_return": pop_returns[i],
                }
            )
    top_above_half = sum(1 for row in top_rows if row["oos_percentile"] >= 50.0)

    pop_returns_finite = [v for v in pop_returns if v is not None]
    beat_bh = (
        sum(1 for v in pop_returns_finite if bh_return is not None and v > bh_return)
        if bh_return is not None
        else None
    )

    report_out = {
        "settings": {
            "population": config.population_size,
            "generations": config.generations,
            "rtr_crowding_factor": args.rtr_cf,
            "seed_rate": args.seed_rate,
            "is_window": [args.is_start, args.is_end],
            "oos_window": [args.oos_start, args.oos_end],
            "ga_elapsed_sec": ga_elapsed,
        },
        "diversity": {
            "final_population": len(population),
            "unique_structures": unique_signatures,
        },
        "is_fitness": summarize(is_primary),
        "oos_fitness_final_pop": summarize(oos_primary),
        "oos_fitness_random_null": summarize(rand_primary),
        "oos_total_return_final_pop": summarize(pop_returns),
        "oos_total_return_random_null": summarize(rand_returns),
        "buy_and_hold_return": bh_return,
        "rank_correlation_is_oos": corr,
        "mannwhitney_evolved_gt_random_p": mwu_p,
        "top_is_strategies": {
            "k": top_k,
            "rows": top_rows,
            "oos_above_median_count": top_above_half,
        },
        "final_pop_beating_buyhold_count": beat_bh,
        # 再集計用の生配列 (None は評価失敗またはペナルティ)
        "arrays": {
            "is_primary": is_primary,
            "oos_primary": oos_primary,
            "pop_returns": pop_returns,
            "rand_primary": rand_primary,
            "rand_returns": rand_returns,
            "pop_trades": pop_trades,
            "rand_trades": rand_trades,
            "pop_max_dd": pop_max_dd,
        },
    }

    logger.info("=" * 60)
    logger.info("=== エッジ検証結果 ===")
    logger.info(
        "順位相関(IS→OOS): Spearman ρ=%s (p=%s), Kendall τ=%s, bootstrap95%%CI=%s",
        corr["spearman_rho"],
        corr["spearman_p"],
        corr["kendall_tau"],
        corr["bootstrap_ci95"],
    )
    logger.info(
        "OOS fitness: 最終集団 central=%s vs ランダム central=%s (MWU p=%s)",
        summarize(oos_primary).get("median"),
        summarize(rand_primary).get("median"),
        mwu_p,
    )
    logger.info(
        "OOS total_return: 最終集団 central=%s vs ランダム central=%s vs B&H=%s",
        summarize(pop_returns).get("median"),
        summarize(rand_returns).get("median"),
        bh_return,
    )
    logger.info(
        "IS上位%d構造のうち OOS中央値以上: %d 個 / B&H超え: %s 個(最終集団)",
        top_k,
        top_above_half,
        beat_bh,
    )
    logger.info(
        "ペナルティ/評価失敗: IS=%d, OOS=%d, ランダムOOS=%d",
        sum(1 for v in is_primary if v is None),
        sum(1 for v in oos_primary if v is None),
        sum(1 for v in rand_primary if v is None),
    )

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report_out, fh, ensure_ascii=False, indent=2)
    logger.info("結果JSON: %s", out_path.resolve())


if __name__ == "__main__":
    main()
