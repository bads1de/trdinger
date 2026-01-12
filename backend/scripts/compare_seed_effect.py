"""
シード戦略あり/なしのGA性能比較スクリプト

簡易的なGAを実行し、シード戦略の効果を検証します。
"""

import time
import logging
from typing import Dict, Any

# 警告を抑制
import warnings

# ログ設定（INFOのみ表示）
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# GAエンジンのログだけINFOで出力
logging.getLogger("app.services.auto_strategy.core.ga_engine").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


warnings.filterwarnings("ignore")


def run_single_ga(use_seeds: bool, seed_rate: float = 0.2) -> Dict[str, Any]:
    """単一のGA実行"""
    from app.services.auto_strategy.config.ga import GAConfig
    from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
    from app.services.auto_strategy.generators.random_gene_generator import (
        RandomGeneGenerator,
    )
    from app.services.backtest.backtest_service import BacktestService

    # 軽量設定
    config = GAConfig(
        population_size=20,
        generations=3,
        crossover_rate=0.7,
        mutation_rate=0.2,
        elite_size=2,
        max_indicators=3,
        min_indicators=1,
        enable_parallel_evaluation=False,
        use_seed_strategies=use_seeds,
        seed_injection_rate=seed_rate if use_seeds else 0.0,
        fitness_constraints={"min_trades": 1},
    )

    backtest_config = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",
        "initial_capital": 10000,
    }

    gene_generator = RandomGeneGenerator(config)
    engine = GeneticAlgorithmEngine(BacktestService(), gene_generator)

    start_time = time.time()
    result = engine.run_evolution(config, backtest_config)
    elapsed = time.time() - start_time

    return {
        "elapsed_time": elapsed,
        "best_fitness": result.get("best_fitness", 0),
        "success": True,
    }


def main():
    print("\n" + "=" * 60)
    print("     シード戦略効果検証スクリプト")
    print("=" * 60)

    results = {}

    # 1. シードなし
    print("\n【実験1】シードなしでGA実行...")
    try:
        results["no_seed"] = run_single_ga(use_seeds=False)
        print(
            f"  完了: {results['no_seed']['elapsed_time']:.1f}秒, "
            f"フィットネス: {results['no_seed']['best_fitness']:.4f}"
        )
    except Exception as e:
        import traceback

        print(f"  エラー: {e}")
        traceback.print_exc()
        results["no_seed"] = {"success": False, "error": str(e)}

    # 2. シードあり
    print("\n【実験2】シードありでGA実行 (注入率: 30%)...")
    try:
        results["with_seed"] = run_single_ga(use_seeds=True, seed_rate=0.3)
        print(
            f"  完了: {results['with_seed']['elapsed_time']:.1f}秒, "
            f"フィットネス: {results['with_seed']['best_fitness']:.4f}"
        )
    except Exception as e:
        print(f"  エラー: {e}")
        results["with_seed"] = {"success": False, "error": str(e)}

    # 結果比較
    print("\n" + "=" * 60)
    print("                      結果比較")
    print("=" * 60)

    if results.get("no_seed", {}).get("success") and results.get("with_seed", {}).get(
        "success"
    ):
        no_seed = results["no_seed"]["best_fitness"]
        with_seed = results["with_seed"]["best_fitness"]

        if no_seed != 0:
            improvement = (with_seed - no_seed) / abs(no_seed) * 100
        else:
            improvement = 0

        print(f"\n{'項目':<25} {'シードなし':>12} {'シードあり':>12}")
        print("-" * 55)
        print(f"{'ベストフィットネス':<25} {no_seed:>12.4f} {with_seed:>12.4f}")
        print(
            f"{'実行時間(秒)':<25} {results['no_seed']['elapsed_time']:>12.1f} "
            f"{results['with_seed']['elapsed_time']:>12.1f}"
        )
        print("-" * 55)
        print(f"{'改善率':<25} {improvement:>+12.1f}%")
        print("=" * 60)

        if with_seed > no_seed:
            print("\n[SUCCESS] シード戦略の効果が確認されました！")
        elif with_seed == no_seed:
            print(
                "\n[INFO] 今回はシード戦略の効果が見られませんでした（ランダム性あり）。"
            )
        else:
            print("\n[WARN] シードなしの方が良い結果となりました。")

    else:
        print("\n[ERROR] 実験に失敗しました。")


if __name__ == "__main__":
    main()
