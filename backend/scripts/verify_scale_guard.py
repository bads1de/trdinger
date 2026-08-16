"""
スケールガード検証スクリプト

GA を実行し、最終集団の全個体について「価格オペランド × 非価格スケール指標」
の退化条件（恒真/恒偽）が発生していないことを検証する。
生成器のスケールガードと repair_condition_scales の効果確認に使う。

使用例:
    uv run python scripts/verify_scale_guard.py --population 50 --generations 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.auto_strategy.config.ga_config import (  # noqa: E402
    EvaluationConfig,
    FitnessSharingConfig,
    GAConfig,
)
from app.services.auto_strategy.core import GeneticAlgorithmEngine  # noqa: E402
from app.services.auto_strategy.generators.random_gene_generator import (  # noqa: E402
    RandomGeneGenerator,
)
from app.services.auto_strategy.genes.strategy import (
    _operand_reference_name,  # noqa: E402
)
from app.services.auto_strategy.utils.indicator_references import (  # noqa: E402
    build_indicator_reference_name,
)
from app.services.auto_strategy.utils.scale_compat import (  # noqa: E402
    is_close_comparable_type,
    is_price_operand,
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


def violations_of(gene: object) -> list[str]:
    """遺伝子内の全条件を走査し、価格×非価格スケール比較を返す。"""
    name_to_type: dict[str, str] = {}
    for ind in gene.indicators:  # type: ignore[attr-defined]
        name_to_type.setdefault(ind.type, ind.type)
        timeframe = getattr(ind, "timeframe", None)
        if timeframe:
            name_to_type.setdefault(f"{ind.type}_{timeframe}", ind.type)
        name_to_type[build_indicator_reference_name(ind)] = ind.type

    def resolve(name: str) -> str | None:
        resolved = name_to_type.get(name)
        if resolved is None:
            stem, _, suffix = name.rpartition("_")
            if stem and suffix.isdigit():
                resolved = name_to_type.get(stem)
        return resolved

    found: list[str] = []

    def scan(item: object) -> None:
        if hasattr(item, "conditions"):
            for sub in item.conditions:
                scan(sub)
            return
        for price_attr, other_attr in (
            ("left_operand", "right_operand"),
            ("right_operand", "left_operand"),
        ):
            if not is_price_operand(getattr(item, price_attr, None)):
                continue
            other_name = _operand_reference_name(getattr(item, other_attr))
            if other_name is None:
                continue
            try:
                float(other_name)
                continue  # 数値リテラル
            except ValueError:
                pass
            other_type = resolve(other_name)
            if other_type and not is_close_comparable_type(other_type):
                found.append(f"{getattr(item, price_attr)} {other_name} [{other_type}]")

    for conditions in (
        gene.long_entry_conditions,  # type: ignore[attr-defined]
        gene.short_entry_conditions,  # type: ignore[attr-defined]
        gene.long_exit_conditions,  # type: ignore[attr-defined]
        gene.short_exit_conditions,  # type: ignore[attr-defined]
    ):
        for item in conditions:
            scan(item)
    for stateful in gene.stateful_conditions:  # type: ignore[attr-defined]
        scan(stateful.trigger_condition)
        scan(stateful.follow_condition)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="スケールガードの集団検証")
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed-rates", type=str, default="0.0,0.3")
    parser.add_argument(
        "--mtf", action="store_true", help="マルチタイムフレームを有効化"
    )
    parser.add_argument(
        "--no-sharing",
        action="store_true",
        help="適応度共有を無効化して選抜のみの挙動を計測",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    for rate in [float(r) for r in args.seed_rates.split(",")]:
        use_seeds = rate > 0.0
        label = f"seed_rate={rate}" + (" [no-sharing]" if args.no_sharing else "")
        config = GAConfig(
            population_size=4 if args.smoke else args.population,
            generations=1 if args.smoke else args.generations,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2,
            max_indicators=10,
            max_conditions=3,
            non_price_indicator_probability=0.3,
            use_seed_strategies=use_seeds,
            seed_injection_rate=rate if use_seeds else 0.0,
            enable_multi_timeframe=args.mtf,
            available_timeframes=["1d"] if args.mtf else None,
            mtf_indicator_probability=0.3,
            fitness_sharing=(
                FitnessSharingConfig(enable_fitness_sharing=False)
                if args.no_sharing
                else FitnessSharingConfig()
            ),
            evaluation_config=EvaluationConfig(enable_parallel=not args.smoke),
            fallback_start_date="2024-01-01",
            fallback_end_date="2024-06-30",
        )
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "4h",
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "initial_capital": 100000.0,
            "commission_rate": 0.0004,
            "slippage": 0.0001,
        }

        engine = GeneticAlgorithmEngine(
            backtest_service=BacktestService(),
            gene_generator=RandomGeneGenerator(config=config),
        )
        t0 = time.time()
        result = engine.run_evolution(config=config, backtest_config=backtest_config)
        elapsed = time.time() - t0

        all_strategies = result.get("all_strategies", [])
        genes_with = 0
        total_violations = 0
        types: Counter[str] = Counter()
        for gene in all_strategies:
            found = violations_of(gene)
            if found:
                genes_with += 1
                total_violations += len(found)
                types.update(v.rsplit("[", 1)[-1].rstrip("]") for v in found)

        logger.info("=" * 60)
        logger.info(
            "%s: %.0fs best_fitness=%s population=%d",
            label,
            elapsed,
            result.get("best_fitness"),
            len(all_strategies),
        )
        logger.info(
            "スケール違反: %d/%d 個体, %d 条件, 種類=%s",
            genes_with,
            len(all_strategies),
            total_violations,
            dict(types),
        )

        unique_sets = {
            tuple(
                sorted(
                    {
                        ind.type
                        for ind in getattr(s, "indicators", [])
                        if getattr(ind, "enabled", True)
                    }
                )
            )
            for s in all_strategies
        }
        logger.info("ユニーク指標構成: %d/%d", len(unique_sets), len(all_strategies))

        # 条件パターンの多様性センサス
        mtf_genes = 0
        trend_follow_genes = 0
        leaf_signatures: set[str] = set()

        def collect_leaves(item: object, leaves: list) -> None:
            if hasattr(item, "conditions"):
                for sub in item.conditions:
                    collect_leaves(sub, leaves)
                return
            leaves.append(item)

        def collect_and_groups(item: object, groups: list) -> None:
            """条件ツリーから AND グループを再帰的に収集する"""
            if not hasattr(item, "conditions"):
                return
            if item.operator == "AND":
                groups.append(item)
            for sub in item.conditions:
                collect_and_groups(sub, groups)

        for gene in all_strategies:
            if any(
                getattr(ind, "timeframe", None)
                for ind in gene.indicators  # type: ignore[attr-defined]
            ):
                mtf_genes += 1

            groups: list = []
            for conditions in (
                gene.long_entry_conditions,  # type: ignore[attr-defined]
                gene.short_entry_conditions,  # type: ignore[attr-defined]
            ):
                for item in conditions:
                    collect_and_groups(item, groups)

            overlay_refs = {
                build_indicator_reference_name(ind)
                for ind in gene.indicators  # type: ignore[attr-defined]
                if is_close_comparable_type(ind.type)
            }
            for item in groups:
                if not hasattr(item, "conditions") or item.operator != "AND":
                    continue
                leaves: list = []
                collect_leaves(item, leaves)
                has_price_vs_overlay = any(
                    is_price_operand(getattr(leaf, "left_operand", None))
                    and getattr(leaf, "right_operand", None) in overlay_refs
                    for leaf in leaves
                )
                has_threshold_leaf = any(
                    isinstance(getattr(leaf, "right_operand", None), (int, float))
                    for leaf in leaves
                )
                if has_price_vs_overlay and has_threshold_leaf:
                    trend_follow_genes += 1
                    break

            for conditions in (
                gene.long_entry_conditions,  # type: ignore[attr-defined]
                gene.short_entry_conditions,  # type: ignore[attr-defined]
                gene.long_exit_conditions,  # type: ignore[attr-defined]
                gene.short_exit_conditions,  # type: ignore[attr-defined]
            ):
                leaves = []
                for item in conditions:
                    collect_leaves(item, leaves)
                for leaf in leaves:
                    leaf_signatures.add(
                        f"{leaf.left_operand} {leaf.operator} {leaf.right_operand}"  # type: ignore[attr-defined]
                    )

        logger.info(
            "条件パターン: MTF個体=%d, トレンドフォロー個体=%d, ユニーク葉条件=%d",
            mtf_genes,
            trend_follow_genes,
            len(leaf_signatures),
        )

        # 旧シード（スケールガード導入前の遺伝子）を持ち込んだ場合の
        # 浄化確認: シードあり実行では repair_condition_scales が交叉・
        # 変異後に走るため、最終集団に違反が残らなければOK
        if genes_with > 0:
            logger.error("スケール違反が残存しています: %s", dict(types))
            sys.exit(1)

    logger.info("検証完了: 全実行でスケール違反ゼロ")


if __name__ == "__main__":
    main()
