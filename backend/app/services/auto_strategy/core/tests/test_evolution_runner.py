"""
EvolutionRunnerのユニットテスト

TDDに基づくGAエンジン分割のテストケース。
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np
from deap import base, creator, tools

from app.services.auto_strategy.core.evolution_runner import EvolutionRunner
from app.services.auto_strategy.config.ga_config import GAConfig


class TestEvolutionRunner:
    """EvolutionRunnerのテストクラス"""

    @pytest.fixture
    def mock_toolbox(self):
        """モックされたDEAPツールボックス"""
        toolbox = Mock()
        # 基本的なメソッドを設定
        toolbox.map = lambda func, *args: list(map(func, *args))
        toolbox.clone = lambda x: x
        toolbox.mate = Mock()
        toolbox.mutate = Mock()
        toolbox.evaluate = Mock()
        toolbox.select = Mock(side_effect=lambda pop, n: pop[:n])
        return toolbox

    @pytest.fixture
    def mock_stats(self):
        """モックされた統計オブジェクト"""
        stats = Mock()
        stats.compile = Mock(return_value={"avg": 0.5, "max": 1.0})
        return stats

    @pytest.fixture
    def mock_config(self):
        """テスト用GA設定"""
        config = Mock()
        config.generations = 2
        config.crossover_rate = 0.8
        config.mutation_rate = 0.1
        config.enable_fitness_sharing = False
        return config

    @pytest.fixture
    def mock_population(self):
        """モック個体群"""
        # 個体をモック
        population = []
        for i in range(5):
            individual = Mock()
            individual.fitness = Mock()
            individual.fitness.valid = True
            individual.fitness.values = (0.5,)
            population.append(individual)
        return population

    def test_evolution_runner_initialization(
        self, mock_toolbox, mock_stats
    ):
        """EvolutionRunnerの初期化が正しく行えるかテスト"""
        # 通常の初期化
        runner = EvolutionRunner(mock_toolbox, mock_stats)
        assert runner.toolbox == mock_toolbox
        assert runner.stats == mock_stats
        assert runner.fitness_sharing is None

        # 適応度共有ありの初期化
        mock_fs = Mock()
        runner_with_fs = EvolutionRunner(
            mock_toolbox, mock_stats, fitness_sharing=mock_fs
        )
        assert runner_with_fs.fitness_sharing == mock_fs

    def test_single_objective_evolution_basic(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """単一目的進化の基本的な実行をテスト"""
        # 個体の適応度を無効化してからテスト
        for ind in mock_population:
            ind.fitness.valid = False

        # 評価をモック
        mock_toolbox.evaluate = Mock(side_effect=lambda x: (0.8,))

        runner = EvolutionRunner(mock_toolbox, mock_stats)

        # 実行
        result_pop, logbook = runner.run_single_objective_evolution(
            mock_population, mock_config
        )

        # 基本的な検証
        assert isinstance(result_pop, list)
        assert hasattr(logbook, 'record')

    def test_multi_objective_evolution_basic(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """多目的進化の基本的な実行をテスト"""
        # 個体の適応度を無効化
        for ind in mock_population:
            ind.fitness.valid = False

        # 評価をモック（多目的）
        mock_toolbox.evaluate = Mock(side_effect=lambda x: (0.8, 0.6))

        # NSGA2選択をモック
        def mock_selNSGA2(pop, n):
            return pop[:n]
        
        with patch('deap.tools.selNSGA2', mock_selNSGA2):
            runner = EvolutionRunner(mock_toolbox, mock_stats)
            result_pop, logbook = runner.run_multi_objective_evolution(
                mock_population, mock_config
            )

            # 基本的な検証
            assert isinstance(result_pop, list)
            assert hasattr(logbook, 'record')

    def test_population_evaluation(
        self, mock_toolbox, mock_stats, mock_population
    ):
        """個体群評価が正しく行われるかテスト"""
        # 全個体の適応度を無効化
        for ind in mock_population:
            ind.fitness.valid = False
            ind.fitness.values = ()

        # 評価関数をモック
        def mock_evaluate(x):
            return (0.5,)

        mock_toolbox.map = Mock(side_effect=lambda func, pop: [func(ind) for ind in pop])

        runner = EvolutionRunner(mock_toolbox, mock_stats)
        result_pop = runner._evaluate_population(mock_population)

        # 適応度が設定されているか確認
        for ind in result_pop:
            assert ind.fitness.values == (0.5,)

    @patch('app.services.auto_strategy.core.evolution_runner.logger')
    def test_logging_in_single_objective(
        self, mock_logger, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """単一目的進化のログ出力が正しいかテスト"""
        # 適応度を有効にして初期評価をスキップ
        for ind in mock_population:
            ind.fitness.valid = True

        runner = EvolutionRunner(mock_toolbox, mock_stats)

        runner.run_single_objective_evolution(mock_population, mock_config)

        # ログが呼ばれているか確認
        mock_logger.info.assert_any_call("単一目的最適化アルゴリズムを開始")
        mock_logger.info.assert_any_call("単一目的最適化アルゴリズム完了")

    @patch('app.services.auto_strategy.core.evolution_runner.logger')
    def test_logging_in_multi_objective(
        self, mock_logger, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """多目的進化のログ出力が正しいかテスト"""
        # 適応度を有効にして初期評価をスキップ
        for ind in mock_population:
            ind.fitness.valid = True

        runner = EvolutionRunner(mock_toolbox, mock_stats)

        runner.run_multi_objective_evolution(mock_population, mock_config)

        # ログが呼ばれているか確認
        mock_logger.info.assert_any_call("多目的最適化アルゴリズム（NSGA-II）を開始")
        mock_logger.info.assert_any_call("多目的最適化アルゴリズム（NSGA-II）完了")

    def test_fitness_sharing_integration(
        self, mock_toolbox, mock_stats, mock_population
    ):
        """適応度共有の統合が正しく行われるかテスト"""
        # 設定を作成
        config = Mock()
        config.generations = 1
        config.crossover_rate = 0.0  # 交差を無効化
        config.mutation_rate = 0.0   # 突変を無効化
        config.enable_fitness_sharing = True

        # 適応度共有をモック
        mock_fs = Mock()
        mock_fs.apply_fitness_sharing = Mock(return_value=mock_population)

        runner = EvolutionRunner(
            mock_toolbox, mock_stats, fitness_sharing=mock_fs
        )

        # 適応度を有効化
        for ind in mock_population:
            ind.fitness.valid = True

        # 実行
        result_pop, _ = runner.run_single_objective_evolution(
            mock_population, config
        )

        # 適応度共有が呼ばれたか確認
        mock_fs.apply_fitness_sharing.assert_called()

    def test_hall_of_fame_update(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """殿堂入りの更新が正しく行われるかテスト"""
        # 適応度を有効化
        for ind in mock_population:
            ind.fitness.valid = True

        halloffame = Mock()
        halloffame.update = Mock()

        runner = EvolutionRunner(mock_toolbox, mock_stats)

        runner.run_single_objective_evolution(
            mock_population, mock_config, halloffame=halloffame
        )

        # 殿堂入りが更新されたか確認
        halloffame.update.assert_called()
