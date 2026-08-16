"""fitness sharing 実効挙動の診断スクリプト

GA を短く実行し、世代ごとに:
- 選択プールのユニーク指標構成数（共有適用前）
- 共有ペナルティの実効倍率（適用後の適応度 / 生の適応度）の統計量
- 選択後（次世代）のユニーク指標構成数と生の適応度の最大値
を記録する。共有が実際に効いているか（倍率≈1.0 なら無効）と、
選抜で多様性がどこで失われるかを切り分けるために使う。

使用例:
    uv run python scripts/diagnose_sharing.py --population 30 --generations 15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.auto_strategy.config.ga_config import (  # noqa: E402
    EvaluationConfig,
    GAConfig,
)
from app.services.auto_strategy.core import GeneticAlgorithmEngine  # noqa: E402
from app.services.auto_strategy.core.engine.evolution_runner import (  # noqa: E402
    EvolutionRunner,
)
from app.services.auto_strategy.core.fitness.fitness_sharing import (  # noqa: E402
    FitnessSharing,
)
from app.services.auto_strategy.core.fitness.fitness_validation import (  # noqa: E402
    has_valid_fitness,
)
from app.services.auto_strategy.generators.random_gene_generator import (  # noqa: E402
    RandomGeneGenerator,
)
from app.services.backtest.services.backtest_service import (  # noqa: E402
    BacktestService,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("diagnose")
logger.setLevel(logging.INFO)


def unique_indicator_sets(population: list) -> int:
    return len(
        {
            tuple(sorted({ind.type for ind in getattr(s, "indicators", [])}))
            for s in population
        }
    )


def primary_fitness(ind) -> float | None:
    fitness = getattr(ind, "fitness", None)
    values = getattr(fitness, "values", None)
    if not values:
        return None
    value = float(values[0])
    return value if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="fitness sharing 診断")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=15)
    args = parser.parse_args()

    generation_log: list[dict] = []

    original_apply = FitnessSharing.apply_fitness_sharing

    def instrumented_apply(self, population: list) -> list:
        raw_values = {}
        for i, ind in enumerate(population):
            if has_valid_fitness(ind):
                value = primary_fitness(ind)
                if value is not None and value != 0.0:
                    raw_values[i] = value

        result = original_apply(self, population)

        ratios = []
        for i, raw in raw_values.items():
            shared = primary_fitness(population[i])
            if shared is not None and np.isfinite(shared):
                ratios.append(shared / raw)
        pool_unique = unique_indicator_sets(population)

        # この呼び出しでどの世代かは呼び出し順から復元できないため、
        # 適用回数を世代インデックスの代理として使う
        entry = generation_log
        entry.append(
            {
                "pool": len(population),
                "pool_unique": pool_unique,
                "ratio_min": min(ratios) if ratios else float("nan"),
                "ratio_median": float(np.median(ratios)) if ratios else float("nan"),
                "ratio_max": max(ratios) if ratios else float("nan"),
                "ratio_lt05": sum(1 for r in ratios if r < 0.5),
                "n_ratios": len(ratios),
            }
        )
        return result

    FitnessSharing.apply_fitness_sharing = instrumented_apply  # type: ignore[method-assign]

    original_shared = EvolutionRunner._apply_shared_selection

    def instrumented_selection(self, candidate_population, population_size, config):
        # 選択完了・適応度復元後の個体群を記録する
        selected = original_shared(self, candidate_population, population_size, config)
        finite = [primary_fitness(ind) for ind in selected]
        finite = [v for v in finite if v is not None]
        if generation_log:
            generation_log[-1].update(
                {
                    "selected_unique": unique_indicator_sets(selected),
                    "selected_best_raw": max(finite) if finite else float("nan"),
                    "selected_size": len(selected),
                }
            )
        return selected

    EvolutionRunner._apply_shared_selection = instrumented_selection  # type: ignore[method-assign]

    config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_rate=0.8,
        mutation_rate=0.2,
        max_indicators=10,
        max_conditions=3,
        non_price_indicator_probability=0.3,
        use_seed_strategies=False,
        seed_injection_rate=0.0,
        evaluation_config=EvaluationConfig(enable_parallel=True),
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
    result = engine.run_evolution(config=config, backtest_config=backtest_config)

    logger.info("=" * 72)
    logger.info(
        "%5s %6s %8s %8s | %8s %8s %8s %6s | %8s %8s",
        "call",
        "pool",
        "poolUniq",
        "selUniq",
        "ratioMin",
        "ratioMed",
        "ratioMax",
        "<0.5",
        "selBest",
        "nRatio",
    )
    for i, entry in enumerate(generation_log):
        logger.info(
            "%5d %6d %8d %8s | %8.3f %8.3f %8.3f %6d | %8s %8d",
            i,
            entry["pool"],
            entry.get("pool_unique", -1),
            entry.get("selected_unique", "-"),
            entry.get("ratio_min", float("nan")),
            entry.get("ratio_median", float("nan")),
            entry.get("ratio_max", float("nan")),
            entry.get("ratio_lt05", -1),
            (
                f"{entry['selected_best_raw']:.3f}"
                if "selected_best_raw" in entry
                else "-"
            ),
            entry.get("n_ratios", -1),
        )
    logger.info("最終 best_fitness=%s", result.get("best_fitness"))


if __name__ == "__main__":
    main()
