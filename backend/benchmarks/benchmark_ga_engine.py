"""
進化エンジンのベンチマーク

進化エンジンのパフォーマンスを測定します。
"""

import logging
import os
import sys
import time
from unittest.mock import MagicMock

import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_config():
    """モックGA設定を生成"""
    from app.services.auto_strategy.config.ga import GAConfig

    config = GAConfig()
    config.population_size = 20
    config.generations = 5
    config.crossover_rate = 0.8
    config.mutation_rate = 0.1
    config.elite_size = 2
    config.max_indicators = 5
    config.min_indicators = 1
    config.max_conditions = 3
    config.min_conditions = 1
    return config


def benchmark_evolution_runner():
    """EvolutionRunnerのベンチマーク"""
    logger.info("=== EvolutionRunnerベンチマーク ===")

    from app.services.auto_strategy.core.engine.evolution_runner import EvolutionRunner
    from deap import tools

    config = create_mock_config()

    # モックツールボックス
    mock_toolbox = MagicMock()
    mock_toolbox.evaluate.return_value = (0.5,)
    mock_toolbox.select.return_value = []
    mock_toolbox.mate.side_effect = lambda c1, c2: (c1, c2)
    mock_toolbox.mutate.side_effect = lambda c: (c,)
    mock_toolbox.clone = lambda x: x

    # モック統計
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)

    # モックパラレル評価器
    mock_parallel_evaluator = MagicMock()

    # モック個体クラス
    class MockIndividual:
        def __init__(self):
            self.fitness = MagicMock()
            self.fitness.values = (0.5,)
            self.fitness.valid = True

    # モック集団
    population = [MockIndividual() for _ in range(config.population_size)]

    runner = EvolutionRunner(
        toolbox=mock_toolbox,
        stats=stats,
        population=population,
        parallel_evaluator=mock_parallel_evaluator,
    )

    # ウームアップ
    for _ in range(3):
        mock_toolbox.select(population, config.population_size)
        runner._apply_crossover_batch(population, config)
        runner._apply_mutation_batch(population, config)

    # ベンチマーク実行
    n_iterations = 100

    # 選択演算子
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        mock_toolbox.select(population, config.population_size)
    elapsed_select = time.perf_counter() - start_time

    logger.info(f"\n=== 結果 ({n_iterations}回実行) ===")
    logger.info(
        f"選択演算子: {elapsed_select:.4f}秒 ({elapsed_select/n_iterations*1000:.3f}ms/回)"
    )

    # 交叉演算子
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        runner._apply_crossover_batch(population, config)
    elapsed_crossover = time.perf_counter() - start_time

    logger.info(
        f"交叉演算子: {elapsed_crossover:.4f}秒 ({elapsed_crossover/n_iterations*1000:.3f}ms/回)"
    )

    # 突然変異演算子
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        runner._apply_mutation_batch(population, config)
    elapsed_mutation = time.perf_counter() - start_time

    logger.info(
        f"突然変異演算子: {elapsed_mutation:.4f}秒 ({elapsed_mutation/n_iterations*1000:.3f}ms/回)"
    )

    return {
        "select_ms": elapsed_select / n_iterations * 1000,
        "crossover_ms": elapsed_crossover / n_iterations * 1000,
        "mutation_ms": elapsed_mutation / n_iterations * 1000,
    }


def benchmark_fitness_sharing():
    """FitnessSharingのベンチマーク"""
    logger.info("\n=== FitnessSharingベンチマーク ===")

    from app.services.auto_strategy.core.fitness.fitness_sharing import FitnessSharing
    from app.services.auto_strategy.config.ga import GAConfig

    config = GAConfig()
    config.enable_fitness_sharing = True
    config.sharing_radius = 0.1
    config.sharing_alpha = 1.0

    sharing = FitnessSharing(
        sharing_radius=config.sharing_radius, alpha=config.sharing_alpha
    )

    # モック個体
    class MockIndividual:
        def __init__(self, values):
            self.fitness = MagicMock()
            self.fitness.values = values
            self.fitness.valid = True
            self._feature_vector = np.random.random(20)  # 近似ベクトル

    population = [MockIndividual((np.random.random(),)) for _ in range(20)]

    # ウームアップ
    for _ in range(3):
        sharing.apply_fitness_sharing(population)

    # ベンチマーク実行
    n_iterations = 100

    start_time = time.perf_counter()
    for _ in range(n_iterations):
        sharing.apply_fitness_sharing(population)
    elapsed = time.perf_counter() - start_time

    logger.info(f"\n=== 結果 ({n_iterations}回実行) ===")
    logger.info(
        f"フィットネスシェアリング: {elapsed:.4f}秒 ({elapsed/n_iterations*1000:.3f}ms/回)"
    )

    return {
        "fitness_sharing_ms": elapsed / n_iterations * 1000,
    }


def run_all_benchmarks():
    """全ベンチマークを実行"""
    logger.info("=== 進化エンジン最適化ベンチマーク開始 ===")

    all_results = {}

    try:
        all_results["evolution_runner"] = benchmark_evolution_runner()
    except Exception as e:
        logger.error(f"EvolutionRunnerベンチマークエラー: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["fitness_sharing"] = benchmark_fitness_sharing()
    except Exception as e:
        logger.error(f"FitnessSharingベンチマークエラー: {e}")
        import traceback

        traceback.print_exc()

    logger.info("\n=== 全ベンチマーク完了 ===")

    # サマリー
    logger.info("\n=== サマリー ===")
    if "evolution_runner" in all_results:
        result = all_results["evolution_runner"]
        logger.info(f"選択演算子: {result['select_ms']:.3f}ms/回")
        logger.info(f"交叉演算子: {result['crossover_ms']:.3f}ms/回")
        logger.info(f"突然変異演算子: {result['mutation_ms']:.3f}ms/回")
    if "fitness_sharing" in all_results:
        result = all_results["fitness_sharing"]
        logger.info(
            f"フィットネスシェアリング: {result['fitness_sharing_ms']:.3f}ms/回"
        )

    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
