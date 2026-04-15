# ruff: noqa: E402
"""
進化エンジンの詳細テスト
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# プロジェクトルートをパスに追加
# tests/services/auto_strategy/core/engine/ から backend/ まで5階層上
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
sys.path.insert(0, backend_dir)

from app.services.auto_strategy.core.evaluation.evaluation_report import (
    EvaluationReport,
    ScenarioEvaluation,
)
from app.services.auto_strategy.core.engine.evolution_runner import (
    EvolutionRunner,
    EvolutionStoppedError,
)
from app.services.auto_strategy.core.engine.report_selection import (
    get_two_stage_rank,
)


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
    config.fitness_sharing = {"enable_fitness_sharing": False}
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


class TestEvolutionRunnerAdvanced:
    """進化エンジンの詳細テスト"""

    def setup_method(self):
        """テストセットアップ"""
        pass

    def test_initialization(self, mock_toolbox, mock_stats):
        """初期化テスト"""
        runner = EvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        assert runner.toolbox == mock_toolbox
        assert runner.stats == mock_stats
        assert runner.fitness_sharing is None
        assert runner.parallel_evaluator is None

    def test_run_evolution_basic(self, mock_toolbox, mock_stats, mock_config, mock_population):
        """基本的な進化実行テスト"""
        runner = EvolutionRunner(
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
        runner = EvolutionRunner(
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
        runner = EvolutionRunner(
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
        runner = EvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        offspring = runner._apply_crossover_batch(mock_population, mock_config)

        assert offspring is not None
        assert len(offspring) == len(mock_population)

    def test_mutation_batch(self, mock_toolbox, mock_stats, mock_config, mock_population):
        """突然変異バッチ処理テスト"""
        runner = EvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        offspring = runner._apply_mutation_batch(mock_population, mock_config)

        assert offspring is not None
        assert len(offspring) == len(mock_population)

    def test_evaluate_population(self, mock_toolbox, mock_stats, mock_population):
        """個体群評価テスト"""
        runner = EvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        result = runner._evaluate_population(mock_population)

        assert result is not None
        assert len(result) == len(mock_population)

    def test_evaluate_invalid_individuals(self, mock_toolbox, mock_stats, mock_population):
        """未評価個体評価テスト"""
        runner = EvolutionRunner(
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
        """キャッシュクリアテスト（現在は互換性のため空操作）"""
        runner = EvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        # clear_cachesは現在は空操作（外部インターフェース互換のため残されている）
        runner.clear_caches()
        assert True

    def test_get_crossover_cache_key(self, mock_toolbox, mock_stats):
        """交叉キャッシュキーテスト"""
        runner = EvolutionRunner(
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
        runner = EvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        individual = MagicMock()
        individual.id = "ind_1"

        key = runner._get_mutation_cache_key(individual)

        assert key == "ind_1"

    def test_crossover_recomputes_for_same_parent_ids(self, mock_toolbox, mock_stats):
        """同一IDの親でも交叉結果を使い回さないこと"""
        runner = EvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )
        config = SimpleNamespace(crossover_rate=1.0)

        parent1_a = MagicMock()
        parent1_a.id = "parent-a"
        parent1_a.fitness = MagicMock(valid=True, values=(1.0,))
        parent2_a = MagicMock()
        parent2_a.id = "parent-b"
        parent2_a.fitness = MagicMock(valid=True, values=(1.0,))

        parent1_b = MagicMock()
        parent1_b.id = "parent-a"
        parent1_b.fitness = MagicMock(valid=True, values=(1.0,))
        parent2_b = MagicMock()
        parent2_b.id = "parent-b"
        parent2_b.fitness = MagicMock(valid=True, values=(1.0,))

        mock_toolbox.mate.side_effect = [
            (MagicMock(name="child1_first"), MagicMock(name="child2_first")),
            (MagicMock(name="child1_second"), MagicMock(name="child2_second")),
        ]

        with patch("random.random", return_value=0.0):
            runner._apply_crossover_batch([parent1_a, parent2_a], config)
            runner._apply_crossover_batch([parent1_b, parent2_b], config)

        assert mock_toolbox.mate.call_count == 2

    def test_mutation_recomputes_for_same_individual_id(self, mock_toolbox, mock_stats):
        """同一IDの個体でも突然変異結果を使い回さないこと"""
        runner = EvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )
        config = SimpleNamespace(mutation_rate=1.0)

        individual_a = MagicMock()
        individual_a.id = "same-individual"
        individual_a.fitness = MagicMock(valid=True, values=(1.0,))

        individual_b = MagicMock()
        individual_b.id = "same-individual"
        individual_b.fitness = MagicMock(valid=True, values=(1.0,))

        mock_toolbox.mutate.side_effect = [
            (MagicMock(name="mutant_first"),),
            (MagicMock(name="mutant_second"),),
        ]

        with patch("random.random", return_value=0.0):
            runner._apply_mutation_batch([individual_a], config)
            runner._apply_mutation_batch([individual_b], config)

        assert mock_toolbox.mutate.call_count == 2

    def test_dynamic_objective_scalars_emphasize_risk_metrics(
        self, mock_toolbox, mock_stats
    ):
        """最適化版でもリスク指標の動的スカラーが更新されること"""
        runner = EvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
        )

        def build_individual(values):
            individual = MagicMock()
            individual.fitness = MagicMock()
            individual.fitness.valid = True
            individual.fitness.values = values
            return individual

        population = [
            build_individual((0.5, 0.2, 0.15, 0.4)),
            build_individual((0.4, 0.1, 0.05, 0.2)),
        ]
        config = SimpleNamespace(
            dynamic_objective_reweighting=True,
            objectives=[
                "total_return",
                "max_drawdown",
                "ulcer_index",
                "trade_frequency_penalty",
            ],
            objective_dynamic_scalars={},
        )

        runner._update_dynamic_objective_scalars(population, config)

        assert config.objective_dynamic_scalars["max_drawdown"] == pytest.approx(1.15)
        assert config.objective_dynamic_scalars["ulcer_index"] == pytest.approx(1.1)
        assert config.objective_dynamic_scalars["trade_frequency_penalty"] == pytest.approx(
            1.3
        )
        assert config.objective_dynamic_scalars.get("total_return", 1.0) == 1.0

    def test_two_stage_selection_promotes_report_robust_elite(
        self, mock_toolbox, mock_stats
    ):
        """report の通過率が高い候補をエリートへ昇格させること"""
        raw_leader = MagicMock()
        raw_leader.id = "raw-leader"
        raw_leader.fitness = MagicMock(valid=True, values=(0.95,))

        robust_candidate = MagicMock()
        robust_candidate.id = "robust-candidate"
        robust_candidate.fitness = MagicMock(valid=True, values=(0.90,))

        filler = MagicMock()
        filler.id = "filler"
        filler.fitness = MagicMock(valid=True, values=(0.50,))

        mock_toolbox.select.return_value = [raw_leader, filler]
        mock_evaluator = MagicMock()
        mock_evaluator.get_cached_evaluation_report.side_effect = (
            lambda individual: {
                "raw-leader": EvaluationReport.aggregate(
                    mode="walk_forward",
                    objectives=["weighted_score"],
                    aggregate_method="robust",
                    scenarios=[
                        ScenarioEvaluation(
                            name="fold_1", fitness=(1.0,), passed=True
                        ),
                        ScenarioEvaluation(
                            name="fold_2", fitness=(0.1,), passed=False
                        ),
                    ],
                ),
                "robust-candidate": EvaluationReport.aggregate(
                    mode="walk_forward",
                    objectives=["weighted_score"],
                    aggregate_method="robust",
                    scenarios=[
                        ScenarioEvaluation(
                            name="fold_1", fitness=(0.88,), passed=True
                        ),
                        ScenarioEvaluation(
                            name="fold_2", fitness=(0.82,), passed=True
                        ),
                    ],
                ),
                "filler": EvaluationReport.single(
                    mode="single",
                    objectives=["weighted_score"],
                    scenario=ScenarioEvaluation(
                        name="full", fitness=(0.50,), passed=True
                    ),
                ),
            }[individual.id]
        )

        runner = EvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
            individual_evaluator=mock_evaluator,
        )
        config = SimpleNamespace(
            elite_size=1,
            two_stage_selection_config=SimpleNamespace(
                enabled=True,
                elite_count=1,
                candidate_pool_size=3,
                min_pass_rate=0.0,
            ),
        )

        selected = runner._apply_two_stage_selection(
            [raw_leader, robust_candidate, filler],
            population_size=2,
            config=config,
        )

        assert selected[0] is robust_candidate
        assert get_two_stage_rank(robust_candidate) == 0
        assert raw_leader not in selected[:1]

    def test_two_stage_selection_keeps_population_size_with_duplicate_elite(
        self, mock_toolbox, mock_stats
    ):
        """昇格エリートが重複選択されていても個体数を維持すること"""
        robust_candidate = MagicMock()
        robust_candidate.id = "robust-candidate"
        robust_candidate.fitness = MagicMock(valid=True, values=(0.90,))

        raw_leader = MagicMock()
        raw_leader.id = "raw-leader"
        raw_leader.fitness = MagicMock(valid=True, values=(0.95,))

        filler = MagicMock()
        filler.id = "filler"
        filler.fitness = MagicMock(valid=True, values=(0.50,))

        mock_toolbox.select.return_value = [robust_candidate, robust_candidate]
        mock_evaluator = MagicMock()
        mock_evaluator.get_cached_evaluation_report.side_effect = (
            lambda individual: {
                "raw-leader": EvaluationReport.aggregate(
                    mode="walk_forward",
                    objectives=["weighted_score"],
                    aggregate_method="robust",
                    scenarios=[
                        ScenarioEvaluation(name="fold_1", fitness=(1.0,), passed=True),
                        ScenarioEvaluation(name="fold_2", fitness=(0.1,), passed=False),
                    ],
                ),
                "robust-candidate": EvaluationReport.aggregate(
                    mode="walk_forward",
                    objectives=["weighted_score"],
                    aggregate_method="robust",
                    scenarios=[
                        ScenarioEvaluation(name="fold_1", fitness=(0.88,), passed=True),
                        ScenarioEvaluation(name="fold_2", fitness=(0.82,), passed=True),
                    ],
                ),
                "filler": EvaluationReport.single(
                    mode="single",
                    objectives=["weighted_score"],
                    scenario=ScenarioEvaluation(
                        name="full", fitness=(0.50,), passed=True
                    ),
                ),
            }[individual.id]
        )

        runner = EvolutionRunner(
            toolbox=mock_toolbox,
            stats=mock_stats,
            individual_evaluator=mock_evaluator,
        )
        config = SimpleNamespace(
            elite_size=1,
            two_stage_selection_config=SimpleNamespace(
                enabled=True,
                elite_count=1,
                candidate_pool_size=3,
                min_pass_rate=0.0,
            ),
        )

        selected = runner._apply_two_stage_selection(
            [raw_leader, robust_candidate, filler],
            population_size=2,
            config=config,
        )

        assert len(selected) == 2
        assert selected == [robust_candidate, robust_candidate]


class TestEvolutionRunnerWithParallelEvaluator:
    """並列評価器付き進化エンジンのテスト"""

    def test_run_evolution_with_parallel_evaluator(self, mock_toolbox, mock_stats, mock_config, mock_population):
        """並列評価器付き進化実行テスト"""
        mock_parallel_evaluator = MagicMock()
        mock_parallel_evaluator.evaluate_population.return_value = [(0.5,) for _ in range(10)]

        runner = EvolutionRunner(
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
