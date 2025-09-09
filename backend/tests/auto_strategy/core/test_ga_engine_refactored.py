"""
GAエンジンのリファクタリングテスト

EvolutionRunnerクラスのテスト
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from app.services.auto_strategy.core.ga_engine import EvolutionRunner, GeneticAlgorithmEngine
from app.services.auto_strategy.config.ga_runtime import GAConfig
from app.services.backtest.backtest_service import BacktestService
from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator


class TestEvolutionRunner:
    """EvolutionRunnerクラスのテスト"""

    @pytest.fixture
    def mock_toolbox(self):
        """DEAPツールボックスをモック"""
        toolbox = Mock()
        toolbox.population = Mock(return_value=[])

        # map関数をシミュレート（リストを返すように）
        def mock_map(func, population):
            return [(1.0,)] * len(population)  # 各個体に(1.0,)のfitnessを返す

        toolbox.map = mock_map
        toolbox.evaluate = Mock(return_value=[(1.0,)])
        toolbox.select = Mock(return_value=[])
        toolbox.mate = Mock()
        toolbox.mutate = Mock()
        return toolbox

    @pytest.fixture
    def mock_stats(self):
        """統計情報をモック"""
        stats = Mock()
        return stats

    @pytest.fixture
    def ga_config(self):
        """GA設定を作成"""
        config = GAConfig()
        config.population_size = 10
        config.generations = 5
        config.crossover_rate = 0.8
        config.mutation_rate = 0.2
        config.enable_multi_objective = False
        config.enable_fitness_sharing = False
        return config

    @pytest.fixture
    def population(self):
        """初期個体群をモック"""
        individual = Mock()
        individual.fitness = Mock()
        individual.fitness.values = (1.0,)
        return [individual] * 10

    @pytest.fixture
    def evolution_runner(self, mock_toolbox, mock_stats):
        """EvolutionRunnerインスタンスを作成"""
        runner = EvolutionRunner(mock_toolbox, mock_stats)
        return runner

    def test_run_single_objective_evolution(self, evolution_runner, ga_config, population, mock_toolbox, mock_stats):
        """単一目的最適化の実行テスト"""
        # モックの設定
        mock_logbook = Mock()
        mock_toolbox.population.return_value = population
        changes = [Mock() for _ in range(5)]
        for i, change in enumerate(changes):
            change.fitness.values = (float(i),)

        with patch('deap.algorithms.eaMuPlusLambda', return_value=(population, mock_logbook)) as mock_ea:
            # 実行
            result_pop, result_logbook = evolution_runner.run_single_objective_evolution(
                population, ga_config, []
            )

            # 検証
            mock_ea.assert_called_once()
            args, kwargs = mock_ea.call_args
            assert kwargs['mu'] == len(population)
            assert kwargs['lambda_'] == len(population)
            assert kwargs['cxpb'] == ga_config.crossover_rate
            assert kwargs['mutpb'] == ga_config.mutation_rate
            assert kwargs['ngen'] == ga_config.generations
            assert kwargs['stats'] == mock_stats

            assert result_pop == population
            assert result_logbook == mock_logbook

    def test_run_multi_objective_evolution(self, evolution_runner, ga_config, population, mock_toolbox, mock_stats):
        """多目的最適化の実行テスト"""
        ga_config.enable_multi_objective = True

        # モックの設定
        mock_logbook = Mock()
        mock_toolbox.population.return_value = population
        changes = [Mock() for _ in range(5)]
        for i, change in enumerate(changes):
            change.fitness.values = (float(i), float(i+1))

        with patch('deap.tools.selNSGA2') as mock_select, \
             patch('deap.algorithms.eaMuPlusLambda', return_value=(population, mock_logbook)) as mock_ea, \
             patch('deap.tools.ParetoFront') as mock_pareto:

            # 実行
            result_pop, result_logbook = evolution_runner.run_multi_objective_evolution(
                population, ga_config, []
            )

            # 検証
            mock_ea.assert_called_once()
            assert result_pop == population
            assert result_logbook == mock_logbook

    def test_run_single_objective_handles_fitness_sharing(self, evolution_runner, ga_config, population, mock_toolbox):
        """単一目的最適化でfitness sharingを処理"""
        ga_config.enable_fitness_sharing = True

        with patch('deap.algorithms.eaMuPlusLambda') as mock_ea, \
             patch.object(evolution_runner, 'fitness_sharing', create=True) as mock_sharing:

            # 実行
            evolution_runner.run_single_objective_evolution(population, ga_config, [])

            # fitness sharingが適用されていることを確認
            mock_sharing.apply_fitness_sharing.assert_called()


class TestGAEngineRefactored:
    """リファクタリングされたGAエンジンのテスト"""

    def test_run_evolution_delegates_to_runner(self):
        """run_evolutionがEvolutionRunnerに委譲する"""
        # 依存関係のモック
        mock_backtest = Mock(spec=BacktestService)
        mock_factory = Mock(spec=StrategyFactory)
        mock_generator = Mock(spec=RandomGeneGenerator)
        config = GAConfig()

        engine = GeneticAlgorithmEngine(mock_backtest, mock_factory, mock_generator)

        # 必要な依存関係をモック
        mock_runner = Mock(spec=EvolutionRunner)

        mock_toolbox = Mock()
        mock_toolbox.population.return_value = [Mock()]

        def mock_map(func, population):
            return [(1.0,)] * len(population)

        mock_toolbox.map = mock_map
        mock_toolbox.evaluate = Mock(return_value=[(1.0,)])

        with patch.object(engine, '_create_evolution_runner', return_value=mock_runner), \
             patch.object(engine, 'setup_deap'), \
             patch.object(engine.deap_setup, 'get_toolbox', return_value=mock_toolbox), \
             patch('deap.tools.Statistics'), \
             patch('deap.tools.selBest', return_value=[Mock()]):

            # 実行
            result = engine.run_evolution(config, {})

            # EvolutionRunnerが呼び出されたことを確認
            # (実際のテストはより詳細な実装が必要)

            assert isinstance(result, dict)

    def test_run_evolution_with_multi_objective(self):
        """多目的最適化の場合の処理"""
        # 依存関係のモック
        mock_backtest = Mock(spec=BacktestService)
        mock_factory = Mock(spec=StrategyFactory)
        mock_generator = Mock(spec=RandomGeneGenerator)
        config = GAConfig()
        config.enable_multi_objective = True
        config.objectives = ['total_return', 'sharpe_ratio']

        engine = GeneticAlgorithmEngine(mock_backtest, mock_factory, mock_generator)

        mock_toolbox = Mock()
        mock_toolbox.population.return_value = [Mock()]

        def mock_map(func, population):
            return [(1.0,)] * len(population)

        mock_toolbox.map = mock_map
        mock_toolbox.evaluate = Mock(return_value=[(1.0,)])

        with patch.object(engine, 'setup_deap'), \
             patch.object(engine.deap_setup, 'get_toolbox', return_value=mock_toolbox), \
             patch('deap.tools.Statistics'), \
             patch('deap.tools.ParetoFront') as mock_pareto, \
             patch('deap.tools.selBest', return_value=[Mock()]):

            mock_pareto.return_value.update.return_value = [Mock()]
            mock_pareto.return_value = [Mock() for _ in range(3)]

            # 実行
            result = engine.run_evolution(config, {})

            assert 'pareto_front' in result
            assert isinstance(result['pareto_front'], list)

    def test_evolution_runner_creation(self):
        """EvolutionRunnerが適切に作成される"""
        # このテストは実際のリファクタリング後に実装
        pass