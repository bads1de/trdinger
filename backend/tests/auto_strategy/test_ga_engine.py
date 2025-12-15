"""
GA Engineのテストモジュール
"""

from unittest.mock import Mock, patch

import pytest

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.core.individual_evaluator import (
    IndividualEvaluator,
)
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.generators.strategy_factory import (
    StrategyFactory,
)
from app.services.backtest.backtest_service import BacktestService


class TestGeneticAlgorithmEngine:
    """GeneticAlgorithmEngineの初期化とモード切り替えのテスト"""

    @pytest.fixture
    def mock_backtest_service(self):
        """Mock BacktestService"""
        service = Mock(spec=BacktestService)
        service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "total_trades": 100,
        }
        return service

    @pytest.fixture
    def mock_strategy_factory(self):
        """Mock StrategyFactory"""
        return Mock(spec=StrategyFactory)

    @pytest.fixture
    def mock_gene_generator(self):
        """Mock RandomGeneGenerator"""
        generator = Mock(spec=RandomGeneGenerator)
        from app.services.auto_strategy.models import (
            IndicatorGene,
            PositionSizingGene,
            PositionSizingMethod,
            StrategyGene,
            TPSLGene,
        )

        mock_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
            tpsl_gene=TPSLGene(take_profit_pct=0.01, stop_loss_pct=0.005),
            position_sizing_gene=PositionSizingGene(
                method=PositionSizingMethod.FIXED_QUANTITY, fixed_quantity=1000
            ),
            metadata={"generated_by": "Test"},
        )
        generator.generate_random_gene.return_value = mock_gene
        return generator

    def test_standard_mode_initialization(
        self,
        mock_backtest_service,
        mock_strategy_factory,
        mock_gene_generator,
    ):
        """標準GAモードでの初期化を確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
            hybrid_mode=False,
            hybrid_predictor=None,
            hybrid_feature_adapter=None,
        )

        # 標準モードであることを確認
        assert engine.hybrid_mode is False
        assert isinstance(engine.individual_evaluator, IndividualEvaluator)
        assert engine.backtest_service == mock_backtest_service
        assert engine.strategy_factory == mock_strategy_factory
        assert engine.gene_generator == mock_gene_generator

    def test_hybrid_mode_initialization(
        self,
        mock_backtest_service,
        mock_strategy_factory,
        mock_gene_generator,
    ):
        """ハイブリッドGA+MLモードでの初期化を確認"""
        mock_predictor = Mock()
        mock_feature_adapter = Mock()

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
            hybrid_mode=True,
            hybrid_predictor=mock_predictor,
            hybrid_feature_adapter=mock_feature_adapter,
        )

        # ハイブリッドモードであることを確認
        assert engine.hybrid_mode is True
        # HybridIndividualEvaluatorが使用されていることを確認
        assert type(engine.individual_evaluator).__name__ == "HybridIndividualEvaluator"

    def test_engine_components_are_set(
        self,
        mock_backtest_service,
        mock_strategy_factory,
        mock_gene_generator,
    ):
        """エンジンのコンポーネントが正しく設定されることを確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
            hybrid_mode=False,
            hybrid_predictor=None,
            hybrid_feature_adapter=None,
        )

        # 必須コンポーネントが設定されていることを確認
        assert engine.backtest_service is not None
        assert engine.strategy_factory is not None
        assert engine.gene_generator is not None
        assert engine.individual_evaluator is not None
        assert engine.deap_setup is not None

        # 初期状態の確認
        assert engine.is_running is False
        assert engine.individual_class is None  # setup_deap前はNone
        assert engine.fitness_sharing is None  # setup_deap前はNone

    @patch("app.services.auto_strategy.core.ga_engine.EvolutionRunner")
    def test_run_evolution_flow(
        self,
        mock_runner_cls,
        mock_backtest_service,
        mock_strategy_factory,
        mock_gene_generator,
    ):
        """GAエンジンの実行フローを確認"""
        # セットアップ
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
        )

        # Mock DEAP components
        engine.deap_setup = Mock()
        mock_toolbox = Mock()
        mock_toolbox.population.return_value = ["ind1", "ind2"]
        engine.deap_setup.get_toolbox.return_value = mock_toolbox
        engine.deap_setup.get_individual_class.return_value = Mock()

        # Mock EvolutionRunner instance
        mock_runner_instance = mock_runner_cls.return_value
        mock_runner_instance.run_evolution.return_value = (
            ["best_ind"],
            Mock(),
        )  # pop, logbook

        # Mock internal methods to isolate run_evolution logic
        engine._process_results = Mock(
            return_value={"execution_time": 1.0, "best_fitness": 1.0}
        )
        engine._create_statistics = Mock(return_value=Mock())

        # Config mock
        mock_config = Mock()
        mock_config.population_size = 10
        mock_config.generations = 5
        mock_config.enable_parallel_evaluation = False
        mock_config.enable_fitness_sharing = False
        mock_config.enable_multi_objective = False
        mock_config.fallback_start_date = "2024-01-01"
        mock_config.fallback_end_date = "2024-01-31"

        backtest_config = {"symbol": "BTC/USDT:USDT", "timeframe": "1h"}

        # 実行
        result = engine.run_evolution(mock_config, backtest_config)

        # 検証
        assert engine.is_running is False
        engine.deap_setup.setup_deap.assert_called_once()
        mock_toolbox.population.assert_called_with(n=10)
        mock_runner_cls.assert_called_once()
        mock_runner_instance.run_evolution.assert_called_once()
        engine._process_results.assert_called_once()
        assert result["execution_time"] == 1.0

    @patch("app.services.auto_strategy.core.ga_engine.ParallelEvaluator")
    @patch("app.services.auto_strategy.core.ga_engine.EvolutionRunner")
    def test_run_evolution_with_parallel_config(
        self,
        mock_runner_cls,
        mock_parallel_evaluator_cls,
        mock_backtest_service,
        mock_strategy_factory,
        mock_gene_generator,
    ):
        """並列評価設定時の挙動確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
        )

        # Mocks
        engine.deap_setup = Mock()
        mock_toolbox = Mock()
        mock_toolbox.population.return_value = []
        engine.deap_setup.get_toolbox.return_value = mock_toolbox
        engine.deap_setup.get_individual_class.return_value = Mock()
        engine._process_results = Mock(return_value={"execution_time": 0.5})
        engine._create_statistics = Mock()

        # Mock EvolutionRunner instance behavior
        mock_runner_instance = mock_runner_cls.return_value
        mock_runner_instance.run_evolution.return_value = (["pop"], "log")

        # Config with parallel enabled
        mock_config = Mock()
        mock_config.enable_parallel_evaluation = True
        mock_config.max_evaluation_workers = 4
        mock_config.evaluation_timeout = 60.0
        mock_config.population_size = 10
        # Other necessary attrs
        mock_config.enable_fitness_sharing = False
        mock_config.enable_multi_objective = False
        mock_config.fallback_start_date = "2024-01-01"
        mock_config.fallback_end_date = "2024-01-31"

        # 実行
        engine.run_evolution(mock_config, {})

        # ParallelEvaluatorが初期化されたことを確認
        mock_parallel_evaluator_cls.assert_called_once()
        call_args = mock_parallel_evaluator_cls.call_args[1]
        assert call_args["max_workers"] == 4
        assert call_args["timeout_per_individual"] == 60.0

        # RunnnerにParallelEvaluatorが渡されたか確認
        # EvolutionRunner(toolbox, stats, fitness_sharing, population, parallel_evaluator)
        runner_call_args = mock_runner_cls.call_args[0]
        # 5番目の引数がparallel_evaluator
        assert runner_call_args[4] == mock_parallel_evaluator_cls.return_value
