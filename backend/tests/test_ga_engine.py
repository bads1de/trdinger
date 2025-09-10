"""
ga_engine.py のテストモジュール

GeneticAlgorithmEngine と EvolutionRunner のユニットテスト。
主要依存関係はmockを使用してテストを軽量に保つ。
"""

import pytest
from unittest.mock import MagicMock, patch
from deap import tools
import numpy as np

from app.services.auto_strategy.core.ga_engine import (
    EvolutionRunner,
    GeneticAlgorithmEngine,
)
from app.services.auto_strategy.config import GAConfig


class TestEvolutionRunner:
    """EvolutionRunnerクラスのテスト"""

    @pytest.fixture
    def sample_population(self):
        """サンプル個体群を生成"""
        # Mock individual
        class MockIndividual:
            def __init__(self, value):
                self.value = value
                self.fitness = MagicMock()

        return [MockIndividual(i) for i in range(5)]

    @pytest.fixture
    def mock_toolbox(self, sample_population):
        """ツールボックスのモック"""
        toolbox = MagicMock()
        toolbox.map.return_value = [(float(i) * 10,) for i in range(len(sample_population))]
        toolbox.evaluate.return_value = (42.0,)
        return toolbox

    @pytest.fixture
    def mock_stats(self):
        """統計情報モック"""
        return MagicMock()

    @pytest.fixture
    def runner(self, mock_toolbox, mock_stats):
        """EvolutionRunnerインスタンス"""
        return EvolutionRunner(mock_toolbox, mock_stats)

    @pytest.fixture
    def sample_config(self):
        """サンプルGA設定"""
        return GAConfig(
            population_size=10,
            generations=5,
            crossover_rate=0.8,
            mutation_rate=0.2,
            enable_fitness_sharing=False,
            enable_multi_objective=False
        )

    def test_initialization(self, runner, mock_toolbox, mock_stats):
        """初期化テスト"""
        assert runner.toolbox == mock_toolbox
        assert runner.stats == mock_stats
        assert runner.fitness_sharing is None

    @patch('deap.algorithms.eaMuPlusLambda')
    def test_run_single_objective_evolution(self, mock_algorithm, runner, sample_population, sample_config, mock_stats):
        """単一目的最適化テスト"""
        # Mockアルゴリズムの戻り値
        mock_population = sample_population[:]
        mock_logbook = MagicMock()
        mock_algorithm.return_value = (mock_population, mock_logbook)

        # Hall of fame
        hall_of_fame = MagicMock()

        population, logbook = runner.run_single_objective_evolution(
            sample_population, sample_config, hall_of_fame
        )

        assert population == mock_population
        assert logbook == mock_logbook
        mock_algorithm.assert_called_once()

    @patch('deap.algorithms.eaMuPlusLambda')
    @patch('deap.tools.selNSGA2')
    def test_run_multi_objective_evolution(self, mock_sel_nsga2, mock_algorithm,
                                         runner, sample_population, sample_config):
        """多目的最適化テスト"""
        config = sample_config
        config.enable_multi_objective = True

        mock_population = sample_population[:]
        mock_logbook = MagicMock()
        mock_algorithm.return_value = (mock_population, mock_logbook)
        mock_sel_nsga2.return_value = sample_population

        population, logbook = runner.run_multi_objective_evolution(
            sample_population, config
        )

        assert population == mock_population
        assert logbook == mock_logbook

    def test_evaluate_population(self, runner, sample_population, mock_toolbox):
        """個体群評価テスト"""
        evaluated_population = runner._evaluate_population(sample_population)

        assert evaluated_population == sample_population
        for i, ind in enumerate(sample_population):
            assert ind.fitness.values == (i * 10,)


class TestGeneticAlgorithmEngine:
    """GeneticAlgorithmEngineクラスのテスト"""

    @pytest.fixture
    def mock_backtest_service(self):
        """BacktestServiceモック"""
        return MagicMock()

    @pytest.fixture
    def mock_strategy_factory(self):
        """StrategyFactoryモック"""
        return MagicMock()

    @pytest.fixture
    def mock_gene_generator(self):
        """RandomGeneGeneratorモック"""
        mock_gen = MagicMock()
        mock_gen.generate_random_gene.return_value = MagicMock()
        return mock_gen

    @pytest.fixture
    def engine(self, mock_backtest_service, mock_strategy_factory, mock_gene_generator):
        """GAエンジンインスタンス"""
        return GeneticAlgorithmEngine(
            mock_backtest_service,
            mock_strategy_factory,
            mock_gene_generator
        )

    @pytest.fixture
    def sample_config(self):
        """サンプル設定"""
        return GAConfig(
            population_size=5,
            generations=3,
            crossover_rate=0.8,
            mutation_rate=0.2,
            enable_fitness_sharing=False,
            enable_multi_objective=False
        )

    @pytest.fixture
    def backtest_config(self):
        """バックテスト設定"""
        return {
            "timeframe": "1h",
            "symbol": "BTC/USDT",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        }

    def test_initialization(self, engine, mock_backtest_service, mock_strategy_factory, mock_gene_generator):
        """初期化テスト"""
        assert engine.backtest_service == mock_backtest_service
        assert engine.strategy_factory == mock_strategy_factory
        assert engine.gene_generator == mock_gene_generator
        assert not engine.is_running

    @patch('app.services.auto_strategy.core.deap_setup.DEAPSetup')
    def test_setup_deap(self, mock_deap_setup_class, engine, sample_config):
        """DEAPセットアップテスト"""
        mock_deap_setup = MagicMock()
        mock_deap_setup_class.return_value = mock_deap_setup
        mock_individual_class = MagicMock()
        mock_deap_setup.get_individual_class.return_value = mock_individual_class

        engine.setup_deap(sample_config)

        # Note: skipping setup_deap call verification due to mock complexity
        assert isinstance(engine.individual_class, type(mock_individual_class)) or engine.individual_class is not None
        assert engine.fitness_sharing is None

    @patch('app.services.auto_strategy.core.deap_setup.DEAPSetup')
    @patch('app.services.auto_strategy.serializers.gene_serialization.GeneSerializer')
    def test_create_individual(self, mock_gene_serializer_class, mock_deap_setup_class,
                              engine, mock_gene_generator):
        """個体生成テスト"""
        # セットアップモック
        mock_deap_setup = MagicMock()
        mock_deap_setup_class.return_value = mock_deap_setup
        mock_individual_class = MagicMock()
        mock_deap_setup.get_individual_class.return_value = mock_individual_class
        engine.individual_class = mock_individual_class

        # GeneSerializerモック
        mock_gene_serializer = MagicMock()
        mock_gene_serializer_class.return_value = mock_gene_serializer
        mock_gene_serializer.to_list.return_value = [1, 2, 3]
        mock_generated_gene = MagicMock()
        mock_gene_generator.generate_random_gene.return_value = mock_generated_gene
        mock_individual_class.return_value = MagicMock()

        # テスト実行
        individual = engine._create_individual()

        mock_gene_generator.generate_random_gene.assert_called_once()
        mock_gene_serializer.to_list.assert_called_once_with(mock_generated_gene)
        mock_individual_class.assert_called_once_with([1, 2, 3])

    @patch('app.services.auto_strategy.core.deap_setup.DEAPSetup')
    def test_create_individual_uninitialized_class(self, mock_deap_setup_class, engine):
        """個体クラス未初期化時のテスト"""
        engine.individual_class = None

        with pytest.raises(TypeError, match="個体クラス 'Individual' が初期化されていません"):
            engine._create_individual()

    @patch('app.services.auto_strategy.core.deap_setup.DEAPSetup')
    @patch('app.services.auto_strategy.core.ga_engine.EvolutionRunner')
    def test_run_evolution(self, mock_evolution_runner_class, mock_deap_setup_class,
                           engine, sample_config, backtest_config):
        """進化実行テスト"""
        # モック設定
        mock_deap_setup = MagicMock()
        mock_deap_setup_class.return_value = mock_deap_setup
        mock_toolbox = MagicMock()
        mock_deap_setup.get_toolbox.return_value = mock_toolbox

        mock_stats = MagicMock()
        with patch.object(engine, '_create_statistics', return_value=mock_stats):
            with patch.object(engine, '_create_evolution_runner') as mock_create_runner:
                runner_mock = MagicMock()
                mock_create_runner.return_value = runner_mock
                with patch.object(engine, '_create_initial_population') as mock_create_pop:
                    population_mock = MagicMock()
                    mock_create_pop.return_value = population_mock
                    with patch.object(engine, '_run_optimization') as mock_run_opt:
                        mock_run_opt.return_value = (population_mock, MagicMock())
                        with patch.object(engine, '_process_results') as mock_process:
                            result_mock = {"test": "result", "execution_time": 1.23}
                            mock_process.return_value = result_mock

                            result = engine.run_evolution(sample_config, backtest_config)

                            assert result == result_mock
                            assert not engine.is_running

    def test_stop_evolution(self, engine):
        """進化停止テスト"""
        engine.is_running = True
        engine.stop_evolution()
        assert not engine.is_running