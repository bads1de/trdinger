"""
最適化された進化エンジンのテスト
"""

import os
import sys

# プロジェクトルートをパスに追加
# tests/services/auto_strategy/core/engine/ から backend/ まで5階層上
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
sys.path.insert(0, backend_dir)

import pytest
from unittest.mock import MagicMock, Mock

# 直接インポート（ga_engineを経由しない）
import importlib.util
optimized_evolution_runner_path = os.path.join(
    backend_dir, "app", "services", "auto_strategy", "core", "engine", "optimized_evolution_runner.py"
)
spec = importlib.util.spec_from_file_location(
    "optimized_evolution_runner",
    optimized_evolution_runner_path
)
optimized_evolution_runner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimized_evolution_runner_module)

OptimizedEvolutionRunner = optimized_evolution_runner_module.OptimizedEvolutionRunner
EvolutionStoppedError = optimized_evolution_runner_module.EvolutionStoppedError


@pytest.fixture
def mock_toolbox():
    """モックツールボックスのフィクスチャ"""
    toolbox = MagicMock()
    toolbox.clone.return_value = MagicMock()
    toolbox.mate.return_value = (MagicMock(), MagicMock())
    toolbox.mutate.return_value = (MagicMock(),)
    toolbox.select.return_value = [MagicMock() for _ in range(10)]
    toolbox.evaluate.return_value = (0.5,)
    toolbox.map = lambda func, items: [func(item) for item in items]
    return toolbox


@pytest.fixture
def mock_stats():
    """モック統計情報のフィクスチャ"""
    stats = MagicMock()
    stats.compile.return_value = {"avg": 0.5, "std": 0.1, "min": 0.0, "max": 1.0}
    return stats


@pytest.fixture
def mock_config():
    """モック設定のフィクスチャ"""
    config = MagicMock()
    config.generations = 5
    config.objectives = ["total_return"]
    config.crossover_rate = 0.8
    config.mutation_rate = 0.1
    config.enable_fitness_sharing = False
    config.dynamic_objective_reweighting = False
    return config


@pytest.fixture
def mock_population():
    """モック個体群のフィクスチャ"""
    population = []
    for i in range(10):
        ind = MagicMock()
        ind.fitness = MagicMock()
        ind.fitness.valid = True
        ind.fitness.values = (0.5,)
        ind.id = f"ind_{i}"
        population.append(ind)
    return population


class TestOptimizedEvolutionRunner:
    """最適化された進化エンジンのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        pass

    def test_initialization(self, mock_toolbox, mock_stats):
        """初期化テスト"""
        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        assert runner.toolbox == mock_toolbox
        assert runner.stats == mock_stats
        assert runner.fitness_sharing is None
        assert runner.parallel_evaluator is None

    def test_run_evolution_basic(self, mock_toolbox, mock_stats, mock_config, mock_population):
        """基本的な進化実行テスト"""
        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        result_population, logbook = runner.run_evolution(
            population=mock_population,
            config=mock_config,
        )

        assert result_population is not None
        assert logbook is not None

    def test_run_evolution_with_halloffame(self, mock_toolbox, mock_stats, mock_config, mock_population):
        """殿堂入り個体リスト付き進化実行テスト"""
        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        halloffame = MagicMock()
        halloffame.update = MagicMock()

        result_population, logbook = runner.run_evolution(
            population=mock_population,
            config=mock_config,
            halloffame=halloffame,
        )

        assert result_population is not None
        assert halloffame.update.called

    def test_run_evolution_stopped_before_start(self, mock_toolbox, mock_stats, mock_config, mock_population):
        """開始前に停止した場合のテスト"""
        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        def should_stop():
            return True

        with pytest.raises(EvolutionStoppedError):
            runner.run_evolution(
                population=mock_population,
                config=mock_config,
                should_stop=should_stop,
            )

    def test_crossover_batch(self, mock_toolbox, mock_stats, mock_config, mock_population):
        """交叉バッチ処理テスト"""
        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        offspring = runner._apply_crossover_batch(mock_population, mock_config)

        assert offspring is not None
        assert len(offspring) == len(mock_population)

    def test_mutation_batch(self, mock_toolbox, mock_stats, mock_config, mock_population):
        """突然変異バッチ処理テスト"""
        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        offspring = runner._apply_mutation_batch(mock_population, mock_config)

        assert offspring is not None
        assert len(offspring) == len(mock_population)

    def test_evaluate_population(self, mock_toolbox, mock_stats, mock_population):
        """個体群評価テスト"""
        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        result = runner._evaluate_population(mock_population)

        assert result is not None
        assert len(result) == len(mock_population)

    def test_evaluate_invalid_individuals(self, mock_toolbox, mock_stats, mock_population):
        """未評価個体評価テスト"""
        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        # 未評価個体を設定
        for ind in mock_population:
            ind.fitness.valid = False

        runner._evaluate_invalid_individuals(mock_population)

        # 評価が呼ばれたことを確認
        assert mock_toolbox.evaluate.called

    def test_clear_caches(self, mock_toolbox, mock_stats):
        """キャッシュクリアテスト"""
        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        # キャッシュにデータを追加
        runner._crossover_cache["test"] = (MagicMock(), MagicMock())
        runner._mutation_cache["test"] = MagicMock()

        # キャッシュクリア
        runner.clear_caches()

        assert len(runner._crossover_cache) == 0
        assert len(runner._mutation_cache) == 0

    def test_get_crossover_cache_key(self, mock_toolbox, mock_stats):
        """交叉キャッシュキーテスト"""
        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        parent1 = MagicMock()
        parent1.id = "parent1"
        parent2 = MagicMock()
        parent2.id = "parent2"

        key = runner._get_crossover_cache_key(parent1, parent2)

        assert key == "parent1:parent2"

    def test_get_mutation_cache_key(self, mock_toolbox, mock_stats):
        """突然変異キャッシュキーテスト"""
        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        individual = MagicMock()
        individual.id = "ind_1"

        key = runner._get_mutation_cache_key(individual)

        assert key == "ind_1"


class TestOptimizedEvolutionRunnerWithParallelEvaluator:
    """並列評価器付き最適化された進化エンジンのテスト"""

    def test_run_evolution_with_parallel_evaluator(self, mock_toolbox, mock_stats, mock_config, mock_population):
        """並列評価器付き進化実行テスト"""
        mock_parallel_evaluator = MagicMock()
        mock_parallel_evaluator.evaluate_population.return_value = [(0.5,) for _ in range(10)]

        runner = OptimizedEvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
            parallel_evaluator=mock_parallel_evaluator,
        )

        result_population, logbook = runner.run_evolution(
            population=mock_population,
            config=mock_config,
        )

        assert result_population is not None
        assert mock_parallel_evaluator.evaluate_population.called
