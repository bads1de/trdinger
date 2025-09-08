import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine


class TestGeneticAlgorithmEngine:
    """GeneticAlgorithmEngineの単体テスト"""

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        return MagicMock()

    @pytest.fixture
    def mock_strategy_factory(self):
        """モック戦略ファクトリー"""
        return MagicMock()

    @pytest.fixture
    def mock_gene_generator(self):
        """モック遺伝子生成器"""
        generator = MagicMock()
        mock_gene = MagicMock()
        generator.generate_random_gene.return_value = mock_gene
        return generator

    @pytest.fixture
    def mock_ga_config(self):
        """モックGA設定"""
        from backend.app.services.auto_strategy.config.auto_strategy_config import GAConfig
        config = Mock(spec=GAConfig)
        config.enable_fitness_sharing = False
        config.generations = 5
        config.enable_multi_objective = False
        config.population_size = 10
        config.crossover_rate = 0.8
        config.mutation_rate = 0.2
        return config

    @pytest.fixture
    def mock_toolbox(self):
        """モックDEAPツールボックス"""
        toolbox = MagicMock()
        toolbox.population.return_value = [MagicMock() for _ in range(10)]
        toolbox.map.return_value = [[0.5] for _ in range(10)]
        toolbox.select.return_value = toolbox.population.return_value
        return toolbox

    def test_initialization(self, mock_backtest_service, mock_strategy_factory, mock_gene_generator):
        """初期化テスト"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator
        )

        assert engine.backtest_service == mock_backtest_service
        assert engine.strategy_factory == mock_strategy_factory
        assert engine.gene_generator == mock_gene_generator
        assert engine.is_running == False
        assert engine.fitness_sharing is None

    @patch('backend.app.services.auto_strategy.core.ga_engine.DEAPSetup')
    def test_setup_deap(self, mock_deap_setup_class, mock_backtest_service, mock_strategy_factory,
                        mock_gene_generator, mock_ga_config):
        """DEAPセットアップテスト"""
        # モック設定
        mock_deap_setup = MagicMock()
        mock_toolbox = MagicMock()
        mock_individual_class = MagicMock()

        mock_deap_setup.get_toolbox.return_value = mock_toolbox
        mock_deap_setup.get_individual_class.return_value = mock_individual_class
        mock_deap_setup_class.return_value = mock_deap_setup

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator
        )

        engine.setup_deap(mock_ga_config)

        # アサーション
        mock_deap_setup.setup_deap.assert_called_once()
        assert engine.individual_class == mock_individual_class
        mock_ga_config.generations  # 設定アクセスを確認のため

    @patch('backend.app.services.auto_strategy.core.ga_engine.DEAPSetup')
    @patch('backend.app.services.auto_strategy.core.ga_engine.GeneticSerializer')
    @patch('backend.app.services.auto_strategy.core.ga_engine.StrategyGene')
    @patch('backend.app.services.auto_strategy.core.ga_engine.tools')
    @patch('backend.app.services.auto_strategy.core.ga_engine.np')
    def test_run_evolution_single_objective(self, mock_np, mock_tools, mock_strategy_gene_class,
                                            mock_gene_serializer_class, mock_deap_setup_class,
                                            mock_backtest_service, mock_strategy_factory, mock_gene_generator):
        """進化実行テスト - 単一目的最適化"""
        # モック設定
        mock_np.mean.return_value = 0.5
        mock_np.std.return_value = 0.1
        mock_np.min.return_value = 0.0
        mock_np.max.return_value = 1.0

        mock_population = [MagicMock() for _ in range(10)]
        for ind in mock_population:
            ind.fitness.values = [0.5]

        mock_tools.ParetoFront.return_value = MagicMock()
        mock_tools.selBest.return_value = [mock_population[0]]

        # モック個体
        mock_gene_serializer = MagicMock()
        mock_gene = MagicMock()
        mock_gene_serializer.from_list.return_value = mock_gene
        mock_gene_serializer_class.return_value = mock_gene_serializer

        # DEAPモック
        mock_deap_setup = MagicMock()
        mock_toolbox = MagicMock()
        mock_toolbox.population.return_value = mock_population
        mock_toolbox.map.return_value = [[0.5] for _ in range(10)]
        mock_deap_setup.get_toolbox.return_value = mock_toolbox
        mock_deap_setup_class.return_value = mock_deap_setup

        mock_ga_config = MagicMock()
        mock_ga_config.enable_fitness_sharing = False
        mock_ga_config.generations = 5
        mock_ga_config.enable_multi_objective = False
        mock_ga_config.population_size = 10
        mock_ga_config.crossover_rate = 0.8
        mock_ga_config.mutation_rate = 0.2

        backtest_config = {"symbol": "BTC/USDT"}

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator
        )

        # 実行
        result = engine.run_evolution(mock_ga_config, backtest_config)

        # アサーション
        assert "best_strategy" in result
        assert "best_fitness" in result
        assert "population" in result
        assert "logbook" in result
        assert "execution_time" in result
        assert not engine.is_running  # 完了後はFalse

    @patch('backend.app.services.auto_strategy.core.ga_engine.DEAPSetup')
    def test_run_evolution_multi_objective(self, mock_deap_setup_class, mock_backtest_service,
                                           mock_strategy_factory, mock_gene_generator):
        """進化実行テスト - 多目的最適化"""
        # モック設定
        mock_population = [MagicMock() for _ in range(10)]
        for ind in mock_population:
            ind.fitness.values = [0.5, 0.3]

        mock_deap_setup = MagicMock()
        mock_toolbox = MagicMock()
        mock_toolbox.population.return_value = mock_population
        mock_toolbox.map.return_value = [[0.5, 0.3] for _ in range(10)]
        mock_deap_setup.get_toolbox.return_value = mock_toolbox
        mock_deap_setup_class.return_value = mock_deap_setup

        mock_ga_config = MagicMock()
        mock_ga_config.enable_fitness_sharing = False
        mock_ga_config.generations = 5
        mock_ga_config.enable_multi_objective = True
        mock_ga_config.objectives = ["total_return", "max_drawdown"]
        mock_ga_config.population_size = 10
        mock_ga_config.crossover_rate = 0.8
        mock_ga_config.mutation_rate = 0.2

        with patch('backend.app.services.auto_strategy.core.ga_engine.GeneticSerializer') as mock_s:
            serializer_mock = MagicMock()
            serializer_mock.from_list.return_value = MagicMock()
            mock_s.return_value = serializer_mock

            with patch('backend.app.services.auto_strategy.core.ga_engine.tools') as mock_tools:
                pareto_mock = MagicMock()
                pareto_mock.update.return_value = None
                pareto_mock.__iter__.return_value = [mock_population[0], mock_population[1]]
                mock_tools.ParetoFront.return_value = pareto_mock

                engine = GeneticAlgorithmEngine(
                    backtest_service=mock_backtest_service,
                    strategy_factory=mock_strategy_factory,
                    gene_generator=mock_gene_generator
                )

                result = engine.run_evolution(mock_ga_config, {"symbol": "BTC/USDT"})

                assert "pareto_front" in result
                assert "objectives" in result

    def test_stop_evolution(self, mock_backtest_service, mock_strategy_factory, mock_gene_generator):
        """進化停止テスト"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator
        )

        engine.is_running = True
        engine.stop_evolution()

        assert engine.is_running == False

    @patch('backend.app.services.auto_strategy.core.ga_engine.GeneticSerializer')
    @patch('backend.app.services.auto_strategy.core.ga_engine.DEAPSetup')
    def test_create_individual(self, mock_deap_setup_class, mock_gene_serializer_class,
                               mock_backtest_service, mock_strategy_factory, mock_gene_generator):
        """個体生成テスト"""
        # モック設定
        mock_gene_serializer = MagicMock()
        mock_encoded_gene = [0.1, 0.2, 0.3]
        mock_gene_serializer.to_list.return_value = mock_encoded_gene
        mock_gene_serializer_class.return_value = mock_gene_serializer

        mock_deap_setup = MagicMock()
        mock_individual_class = MagicMock()
        mock_individual_class.return_value = MagicMock()
        mock_deap_setup.get_individual_class.return_value = mock_individual_class
        mock_deap_setup_class.return_value = mock_deap_setup

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator
        )

        # DEAPセットアップを実行
        mock_config = MagicMock()
        engine.setup_deap(mock_config)

        # 個体生成実行
        result = engine._create_individual()

        # アサーション
        mock_gene_generator.generate_random_gene.assert_called_once()
        mock_gene_serializer.to_list.assert_called_once()
        mock_individual_class.assert_called_once_with(mock_encoded_gene)
        assert result is not None