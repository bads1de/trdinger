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
    def mock_gene_generator(self):
        """Mock RandomGeneGenerator"""
        generator = Mock(spec=RandomGeneGenerator)
        from app.services.auto_strategy.genes import (
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
        mock_gene_generator,
    ):
        """標準GAモードでの初期化を確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
            hybrid_mode=False
        )

        # 標準モードであることを確認
        assert engine.hybrid_mode is False
        assert isinstance(engine.individual_evaluator, IndividualEvaluator)
        assert engine.gene_generator == mock_gene_generator

    def test_hybrid_mode_initialization(
        self,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """ハイブリッドGA+MLモードでの初期化を確認"""
        mock_predictor = Mock()
        mock_feature_adapter = Mock()

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
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
        mock_gene_generator,
    ):
        """エンジンのコンポーネントが正しく設定されることを確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator
        )

        # 必須コンポーネントが設定されていることを確認
        assert engine.backtest_service is not None
        assert engine.gene_generator is not None
        assert engine.individual_evaluator is not None
        assert engine.deap_setup is not None

    @patch("app.services.auto_strategy.core.ga_engine.EvolutionRunner")
    def test_run_evolution_flow(
        self,
        mock_runner_cls,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """GAエンジンの実行フローを確認"""
        # セットアップ
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )

        # Mock DEAP components
        engine.deap_setup = Mock()
        mock_toolbox = Mock()
        # フィットネス属性を持つモック個体（StrategyGeneを模倣）
        mock_ind = Mock()
        # StrategyGeneであることを判定させるため
        from app.services.auto_strategy.genes import StrategyGene
        mock_ind.__class__ = StrategyGene
        mock_ind.fitness.valid = True
        mock_ind.fitness.values = (1.0,)
        mock_toolbox.population.return_value = [mock_ind]
        engine.deap_setup.get_toolbox.return_value = mock_toolbox
        engine.deap_setup.get_individual_class.return_value = Mock()

        # Mock EvolutionRunner instance
        mock_runner_instance = mock_runner_cls.return_value
        # 戻り値は (population, logbook) の 2 要素
        mock_runner_instance.run_evolution.return_value = (
            [mock_ind],
            Mock()
        )

        # Config mock
        mock_config = Mock()
        mock_config.population_size = 10
        mock_config.generations = 5
        mock_config.enable_parallel_evaluation = False
        mock_config.enable_fitness_sharing = False
        mock_config.enable_multi_objective = False
        mock_config.mutation_rate = 0.1

        backtest_config = {"symbol": "BTCUSDT", "timeframe": "1h"}

        # 実行
        engine.run_evolution(mock_config, backtest_config)

        # 検証
        engine.deap_setup.setup_deap.assert_called_once()
        mock_toolbox.population.assert_called_with(n=10)
        mock_runner_cls.assert_called_once()

    @patch("app.services.auto_strategy.core.ga_engine.ParallelEvaluator")
    @patch("app.services.auto_strategy.core.ga_engine.EvolutionRunner")
    def test_run_evolution_with_parallel_config(
        self,
        mock_runner_cls,
        mock_parallel_evaluator_cls,
        mock_backtest_service,
        mock_gene_generator,
    ):
        """並列評価設定時の挙動確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            gene_generator=mock_gene_generator,
        )

        # Mocks
        engine.deap_setup = Mock()
        mock_toolbox = Mock()
        from app.services.auto_strategy.genes import StrategyGene
        mock_ind = Mock()
        mock_ind.__class__ = StrategyGene
        mock_ind.fitness.valid = True
        mock_ind.fitness.values = (1.0,)
        mock_toolbox.population.return_value = [mock_ind]
        engine.deap_setup.get_toolbox.return_value = mock_toolbox
        engine.deap_setup.get_individual_class.return_value = Mock()

        # Mock EvolutionRunner instance behavior
        mock_runner_instance = mock_runner_cls.return_value
        mock_runner_instance.run_evolution.return_value = ([mock_ind], Mock())

        # Config
        mock_config = Mock()
        mock_config.enable_parallel_evaluation = True
        mock_config.max_evaluation_workers = 4
        mock_config.evaluation_timeout = 60.0
        mock_config.population_size = 10
        mock_config.enable_fitness_sharing = False
        mock_config.enable_multi_objective = False
        mock_config.mutation_rate = 0.1

        # 実行
        engine.run_evolution(mock_config, {})

        # ParallelEvaluatorが初期化されたことを確認
        mock_parallel_evaluator_cls.assert_called_once()

        # start()とshutdown()が呼ばれたことを確認
        mock_instance = mock_parallel_evaluator_cls.return_value
        mock_instance.start.assert_called_once()
        mock_instance.shutdown.assert_called_once()

        # RunnerにParallelEvaluatorが渡されたか確認
        # EvolutionRunner(toolbox, stats, fitness_sharing, population, parallel_evaluator)
        runner_pos_args = mock_runner_cls.call_args[0]
        # 5番目の引数 (インデックス4) が parallel_evaluator
        assert runner_pos_args[4] == mock_instance