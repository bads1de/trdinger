"""
GA Engineのテストモジュール
"""

from unittest.mock import Mock

import pytest

from backend.app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from backend.app.services.auto_strategy.core.individual_evaluator import (
    IndividualEvaluator,
)
from backend.app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from backend.app.services.auto_strategy.generators.strategy_factory import (
    StrategyFactory,
)
from backend.app.services.backtest.backtest_service import BacktestService


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
        from backend.app.services.auto_strategy.models.strategy_models import (
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
