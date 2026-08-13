"""
シード戦略あり/なしの A/B 比較スクリプト

同一条件（個体数・世代数・期間・WFA検証）で
「組み込みシード注入あり」と「シードなし（全ランダム）」の GA を実行し、
以下の指標を比較します。

- WFA 合格戦略数 / 合格率
- 最終集団に占めるシード子孫の割合（多様性指標）
- 最終集団のユニークな指標構成数（多様性指標）
- 検証前のインサンプル最良フィットネス

結果は `results/auto_strategy/seed_ab_*.json` に保存されます。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.auto_strategy.config.ga import GAConfig  # noqa: E402
from app.services.auto_strategy.config.ga.nested_configs import (  # noqa: E402
    EvaluationConfig,
)
from app.services.auto_strategy.core import GeneticAlgorithmEngine  # noqa: E402
from app.services.auto_strategy.generators.random_gene_generator import (  # noqa: E402
    RandomGeneGenerator,
)
from app.services.auto_strategy.genes.genetic_utils import (  # noqa: E402
    GeneticUtils,
)
from app.services.auto_strategy.serializers.serialization import (  # noqa: E402
    GeneSerializer,
)
from app.services.auto_strategy.services.strategy_validation_service import (  # noqa: E402
    StrategyValidationService,
)
from app.services.backtest.services.backtest_service import (  # noqa: E402
    BacktestService,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def count_seed_descendants(strategies: list[Any]) -> tuple[int, list[str]]:
    """最終集団のうち seed_strategy タグを持つ（シード子孫）個体数を数える。"""
    n = 0
    tags: list[str] = []
    for s in strategies:
        meta = getattr(s, "metadata", None) or {}
        tag = meta.get("seed_strategy")
        tags.append(str(tag) if tag else "-")
        if tag:
            n += 1
    return n, tags


def unique_indicator_sets(strategies: list[Any]) -> set[tuple[str, ...]]:
    """最終集団のユニークな指標タイプ構成を集計する。"""
    sets: set[tuple[str, ...]] = set()
    for s in strategies:
        types = tuple(
            sorted(
                {
                    ind.type
                    for ind in getattr(s, "indicators", [])
                    if getattr(ind, "enabled", True)
                }
            )
        )
        sets.add(types)
    return sets


def run_once(
    use_seeds: bool,
    seed_rate: float,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """単一の GA 実行 + WFA 検証を行い、比較指標をまとめる。"""
    label = "with_seed" if use_seeds else "no_seed"
    logger.info("=" * 60)
    logger.info("開始: %s (シード注入率 %.1f%%)", label, seed_rate * 100)
    logger.info("=" * 60)

    population_size = 4 if smoke else 50
    generations = 1 if smoke else 20
    config = GAConfig(
        population_size=population_size,
        generations=generations,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elite_size=2 if not smoke else 1,
        max_indicators=10 if not smoke else 3,
        max_conditions=3 if not smoke else 2,
        non_price_indicator_probability=0.3,
        use_seed_strategies=use_seeds,
        seed_injection_rate=seed_rate if use_seeds else 0.0,
        evaluation_config=EvaluationConfig(enable_parallel=not smoke),
        fallback_start_date="2022-01-01",
        fallback_end_date="2022-12-31",
    )

    backtest_config = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "4h",
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "initial_capital": 100000.0,
        "commission_rate": 0.0004,
        "slippage": 0.0001,
    }

    # 実効注入数 = min(個体数×注入率, 組み込みシード数)
    from app.services.auto_strategy.generators.seed_strategy_factory import (  # noqa: E501
        SeedStrategyFactory,
    )

    num_builtin_seeds = len(SeedStrategyFactory._SEED_MAPPING)
    expected_seed_count = (
        min(int(config.population_size * seed_rate), num_builtin_seeds)
        if use_seeds
        else 0
    )

    backtest_service = BacktestService()
    gene_generator = RandomGeneGenerator(config=config)
    engine = GeneticAlgorithmEngine(
        backtest_service=backtest_service,
        gene_generator=gene_generator,
    )

    # 進化の実行
    t0 = time.time()
    result = engine.run_evolution(config=config, backtest_config=backtest_config)
    ga_elapsed = time.time() - t0
    logger.info("進化完了: %.1f秒", ga_elapsed)

    # === 検証前の多様性メトリクス（all_strategies は検証で絞られる前に計測） ===
    all_strategies = result.get("all_strategies", [])
    seed_count, seed_tags = count_seed_descendants(all_strategies)
    uniq_sets = unique_indicator_sets(all_strategies)

    # 戦略キー -> シードタグ のマップ（検証結果のキーからシード由来か判定するため）
    key_to_seed: dict[str, Any] = {}
    for s in all_strategies:
        meta = getattr(s, "metadata", None) or {}
        try:
            key = GeneticUtils.get_strategy_result_key(s)
        except Exception:  # noqa: BLE001
            key = None
        if key is not None:
            key_to_seed[key] = meta.get("seed_strategy")

    diversity = {
        "population_size": len(all_strategies),
        "seed_descendant_count": seed_count,
        "seed_descendant_fraction": round(seed_count / len(all_strategies), 4)
        if all_strategies
        else None,
        "unique_indicator_set_count": len(uniq_sets),
        "top10_seed_fraction": (
            round(
                sum(1 for t in seed_tags[:10] if t != "-") / min(10, len(seed_tags)),
                4,
            )
            if seed_tags
            else None
        ),
        "seed_tags_top10": seed_tags[:10],
    }

    in_sample_best_fitness = result.get("best_fitness")
    if isinstance(in_sample_best_fitness, tuple):
        in_sample_best_fitness = list(in_sample_best_fitness)

    # === WFA 自動検証 ===
    validation_service = StrategyValidationService(engine.individual_evaluator)
    validated = validation_service.validate_and_filter_result(
        result=result,
        ga_config=config,
        backtest_config=backtest_config,
    )
    vres = validated.get("validation_results", {})
    passed_keys = [k for k, v in vres.items() if v.get("passed", False)]
    logger.info("WFA検証: 対象 %d 件 / 合格 %d 件", len(vres), len(passed_keys))

    # 合格戦略のシードタグと fold サマリを収集
    passed_details: list[dict[str, Any]] = []
    for key in passed_keys:
        info = vres[key]
        passed_details.append(
            {
                "key": key,
                "seed_tag": key_to_seed.get(key),
                "pass_rate": info.get("pass_rate"),
                "primary_fitness": info.get("primary_fitness"),
                "scenario_count": info.get("scenario_count"),
            }
        )

    # 最良戦略（検証後）のシードタグ
    best_strategy = validated.get("best_strategy")
    best_seed_tag = None
    best_gene_dict = None
    if best_strategy is not None:
        meta = getattr(best_strategy, "metadata", None) or {}
        best_seed_tag = meta.get("seed_strategy")
        try:
            best_gene_dict = GeneSerializer().strategy_gene_to_dict(best_strategy)
        except Exception as exc:  # noqa: BLE001
            logger.warning("最良戦略のシリアライズに失敗: %s", exc)

    return {
        "label": label,
        "use_seeds": use_seeds,
        "seed_injection_rate": seed_rate,
        "expected_seed_count": expected_seed_count,
        "ga_elapsed_seconds": round(ga_elapsed, 1),
        "in_sample_best_fitness": in_sample_best_fitness,
        "diversity": diversity,
        "validation": {
            "candidates_validated": len(vres),
            "passed_count": len(passed_keys),
            "passed_details": passed_details,
            "all_results": vres,
        },
        "best_seed_tag": best_seed_tag,
        "best_gene": best_gene_dict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="シード注入率別 A/B 比較（複数レートを同一条件で実行）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="結果を保存するJSONファイルパス",
    )
    parser.add_argument(
        "--seed-rates",
        type=str,
        default="0.0,0.3",
        help="比較するシード注入率（カンマ区切り, 例: 0.0,0.1,0.3）",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="小規模構成（個体4・世代1・並列なし）でパイプライン動作を確認する",
    )
    args = parser.parse_args()

    try:
        seed_rates = [float(r.strip()) for r in args.seed_rates.split(",") if r.strip()]
    except ValueError as exc:
        parser.error(f"--seed-rates の指定が不正です: {exc}")
    for rate in seed_rates:
        if not 0.0 <= rate <= 1.0:
            parser.error(f"注入率は 0.0-1.0 の範囲である必要があります: {rate}")

    results: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "smoke": args.smoke,
        "seed_rates": seed_rates,
        "config": {
            "population_size": 4 if args.smoke else 50,
            "generations": 1 if args.smoke else 20,
            "symbol": "BTC/USDT:USDT",
            "timeframe": "4h",
            "period": "2022-01-01 ~ 2022-12-31",
            "validation": "WFA 5fold min_pass_rate=0.5",
            "non_price_indicator_probability": 0.3,
        },
    }

    for rate in seed_rates:
        key = f"rate_{rate}"
        use_seeds = rate > 0.0
        try:
            results[key] = run_once(
                use_seeds=use_seeds, seed_rate=rate, smoke=args.smoke
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s 実行に失敗: %s", key, exc)
            results[key] = {"label": key, "error": str(exc)}

    # 保存
    output_path = (
        Path(args.output)
        if args.output
        else project_root
        / "results"
        / "auto_strategy"
        / f"seed_ab_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("A/B結果を保存しました: %s", output_path)

    print("\n" + "=" * 60)
    print("シード注入率別 A/B 比較サマリ")
    print("=" * 60)
    for rate in seed_rates:
        r = results.get(f"rate_{rate}", {})
        if "error" in r:
            print(f"\n[rate_{rate}] 失敗: {r['error']}")
            continue
        div = r.get("diversity", {})
        val = r.get("validation", {})
        print(f"\n[rate_{rate}] (注入シード {r.get('expected_seed_count')}個)")
        print(f"  実行時間: {r.get('ga_elapsed_seconds')}秒")
        print(f"  インサンプル最良フィットネス: {r.get('in_sample_best_fitness')}")
        print(
            f"  シード子孫率: {div.get('seed_descendant_fraction')} "
            f"({div.get('seed_descendant_count')}/{div.get('population_size')})"
        )
        print(f"  ユニーク指標構成数: {div.get('unique_indicator_set_count')}")
        print(f"  WFA合格: {val.get('passed_count')}/{val.get('candidates_validated')}")


if __name__ == "__main__":
    sys.exit(main())
