"""
GA Engineのテストモジュール

Regime-aware評価機能とハイブリッドモードをテストする。
"""

from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app.services.auto_strategy.config.ga_runtime import GAConfig
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
from backend.app.services.auto_strategy.services.regime_detector import RegimeDetector
from backend.app.services.backtest.backtest_service import BacktestService


class TestRegimeAwareEvaluation:
    """レジーム対応評価のテスト"""

    @pytest.fixture
    def mock_backtest_service(self):
        """Mock BacktestService"""
        service = Mock(spec=BacktestService)
        # バックテスト結果のモック
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
        factory = Mock(spec=StrategyFactory)
        return factory

    @pytest.fixture
    def mock_gene_generator(self, ga_config_with_regime):
        """Mock RandomGeneGenerator"""
        generator = Mock(spec=RandomGeneGenerator)
        # generate_random_gene メソッドをモック
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

    @pytest.fixture
    def mock_regime_detector(self):
        """Mock RegimeDetector"""
        detector = Mock(spec=RegimeDetector)
        # レジーム検知結果のモック: トレンド、レジーム、高ボラの混合
        detector.detect_regimes.return_value = np.array(
            [0, 1, 2, 0, 1]
        )  # 0=trend, 1=range, 2=high_vol
        return detector

    @pytest.fixture
    def mock_data_service(self):
        """Mock DataService"""
        service = Mock()
        # OHLCVデータのモック
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
                "open": np.random.randn(100) + 100,
                "high": np.random.randn(100) + 101,
                "low": np.random.randn(100) + 99,
                "close": np.random.randn(100) + 100,
                "volume": np.random.randint(1000, 10000, 100),
            }
        )
        service.get_ohlcv_data.return_value = data
        return service

    @pytest.fixture
    def ga_config_with_regime(self):
        """レジーム対応有効のGAConfig"""
        config = GAConfig(
            population_size=50,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            enable_multi_objective=False,
            enable_fitness_sharing=False,
            fallback_start_date="2023-01-01",
            fallback_end_date="2023-12-31",
            objectives=["sharpe_ratio"],
            regime_adaptation_enabled=True,
            max_indicators=5,
            min_indicators=1,
            min_conditions=1,
            max_conditions=3,
        )
        return config

    @pytest.fixture
    def ga_config_without_regime(self):
        """レジーム対応無効のGAConfig"""
        config = GAConfig(
            population_size=50,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            enable_multi_objective=False,
            enable_fitness_sharing=False,
            fallback_start_date="2023-01-01",
            fallback_end_date="2023-12-31",
            objectives=["sharpe_ratio"],
            regime_adaptation_enabled=False,
            max_indicators=5,
            min_indicators=1,
            min_conditions=1,
            max_conditions=3,
        )
        return config

    @pytest.fixture
    def mock_ga_engine(
        self,
        mock_backtest_service,
        mock_strategy_factory,
        mock_gene_generator,
        mock_regime_detector,
    ):
        """Mock GeneticAlgorithmEngine（標準GAモード）"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
            regime_detector=mock_regime_detector,
            hybrid_mode=False,
            hybrid_predictor=None,
            hybrid_feature_adapter=None,
        )
        return engine

    def test_regime_detector_called_when_enabled(
        self,
        mock_backtest_service,
        mock_regime_detector,
        mock_data_service,
        ga_config_with_regime,
    ):
        """regime_adaptation_enabled=Trueの場合、RegimeDetectorが呼ばれることを確認"""
        # mock_backtest_serviceにdata_serviceを設定
        mock_backtest_service.data_service = mock_data_service
        # IndividualEvaluatorの拡張インスタンスを作成（RegimeDetector注入済み）
        evaluator = IndividualEvaluator(mock_backtest_service, mock_regime_detector)

        # 個体のモック
        individual = [0.1, 0.2, 0.3]  # 遺伝子リスト

        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1d",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        evaluator.set_backtest_config(backtest_config)

        # evaluate_individual実行
        with patch.object(mock_data_service, "get_ohlcv_data") as mock_get_data:
            mock_get_data.return_value = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
                    "open": [100] * 100,
                    "high": [101] * 100,
                    "low": [99] * 100,
                    "close": [100] * 100,
                    "volume": [1000] * 100,
                }
            )

            fitness = evaluator.evaluate_individual(individual, ga_config_with_regime)

            # RegimeDetector.detect_regimesが呼ばれたことを確認
            mock_regime_detector.detect_regimes.assert_called_once()
            # データ取得が呼ばれたことを確認
            mock_get_data.assert_called_once_with(
                "BTCUSDT", "1d", "2023-01-01", "2023-12-31"
            )

            # フィットネス値が返されることを確認
            assert isinstance(fitness, tuple)
            assert len(fitness) == 1

    def test_regime_detector_not_called_when_disabled(
        self,
        mock_backtest_service,
        mock_regime_detector,
        mock_data_service,
        ga_config_without_regime,
    ):
        """regime_adaptation_enabled=Falseの場合、RegimeDetectorが呼ばれないことを確認"""
        mock_backtest_service.data_service = mock_data_service
        evaluator = IndividualEvaluator(mock_backtest_service, mock_regime_detector)

        individual = [0.1, 0.2, 0.3]

        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1d",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        evaluator.set_backtest_config(backtest_config)

        # evaluate_individual実行
        fitness = evaluator.evaluate_individual(individual, ga_config_without_regime)

        # RegimeDetectorが呼ばれていないことを確認
        mock_regime_detector.detect_regimes.assert_not_called()

        # フィットネス値が返されることを確認
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1

    def test_regime_based_fitness_adjustment(
        self,
        mock_backtest_service,
        mock_regime_detector,
        mock_data_service,
        ga_config_with_regime,
    ):
        """レジームに基づいてフィットネスが調整されることを確認"""
        mock_backtest_service.data_service = mock_data_service
        evaluator = IndividualEvaluator(mock_backtest_service, mock_regime_detector)

        individual = [0.1, 0.2, 0.3]

        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1d",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        evaluator.set_backtest_config(backtest_config)

        # バックテスト結果にレジーム情報を追加（モック）
        backtest_result_with_regimes = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "total_trades": 100,
            "regime_labels": [0, 1, 2] * 33 + [0],  # トレンド、レンジ、高ボラの混合
            "regime_performance": {
                0: {"sharpe_ratio": 2.0},  # トレンド時はSharpeが高い
                1: {"sharpe_ratio": 1.2},  # レジ時は低い
                2: {"sharpe_ratio": 1.8},  # 高ボラ時は中間
            },
        }
        mock_backtest_service.run_backtest.return_value = backtest_result_with_regimes

        # レジーム検知結果
        regimes = np.array([0] * 50 + [1] * 30 + [2] * 20)  # トレンド多め
        mock_regime_detector.detect_regimes.return_value = regimes

        with patch.object(mock_data_service, "get_ohlcv_data") as mock_get_data:
            mock_get_data.return_value = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
                    "open": [100] * 100,
                    "high": [101] * 100,
                    "low": [99] * 100,
                    "close": [100] * 100,
                    "volume": [1000] * 100,
                }
            )

            fitness = evaluator.evaluate_individual(individual, ga_config_with_regime)

            # フィットネスが計算され、レジーム考慮されていることを確認
            assert isinstance(fitness, tuple)
            assert len(fitness) == 1
            assert isinstance(fitness[0], (int, float))

    def test_error_handling_regime_detection_failure(
        self,
        mock_backtest_service,
        mock_regime_detector,
        mock_data_service,
        ga_config_with_regime,
    ):
        """レジーム検知失敗時のエラーハンドリングを確認"""
        mock_backtest_service.data_service = mock_data_service
        evaluator = IndividualEvaluator(mock_backtest_service, mock_regime_detector)

        individual = [0.1, 0.2, 0.3]

        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1d",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        evaluator.set_backtest_config(backtest_config)

        # レジーム検知でエラーを発生
        mock_regime_detector.detect_regimes.side_effect = Exception(
            "Regime detection failed"
        )

        # evaluate_individual実行 - エラーが発生してもフォールバックされるはず
        fitness = evaluator.evaluate_individual(individual, ga_config_with_regime)

        # デフォルトフィットネスが返されることを確認（取引回数0のため0.1）
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert fitness[0] == 0.1  # 取引回数0時のデフォルト値


class TestRiskFocusedObjectives:
    """リスク指標強化に関する評価ロジックのテスト"""

    def test_multi_objective_returns_ulcer_and_trade_penalty(self) -> None:
        """多目的フィットネスにulcer indexと取引頻度ペナルティを含める。"""

        evaluator = IndividualEvaluator(Mock(spec=BacktestService))

        config = GAConfig(
            enable_multi_objective=True,
            objectives=[
                "total_return",
                "ulcer_index",
                "trade_frequency_penalty",
            ],
            objective_weights=[1.0, -1.0, -1.0],
        )

        start = "2024-01-01"
        end = "2024-01-11"
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.35,
                "sharpe_ratio": 1.1,
                "max_drawdown": 0.2,
                "total_trades": 40,
            },
            "equity_curve": [
                {"timestamp": start, "equity": 100000, "drawdown": 0.0},
                {"timestamp": "2024-01-03", "equity": 99500, "drawdown": 0.05},
                {"timestamp": "2024-01-07", "equity": 99000, "drawdown": 0.1},
                {"timestamp": end, "equity": 102000, "drawdown": 0.0},
            ],
            "trade_history": [{"entry_time": start}] * 40,
            "start_date": start,
            "end_date": end,
        }

        fitness_values = evaluator._calculate_multi_objective_fitness(
            backtest_result,
            config,
        )

        assert len(fitness_values) == 3
        assert fitness_values[0] == pytest.approx(0.35, rel=1e-6)
        assert fitness_values[1] == pytest.approx(0.0559, rel=1e-3)
        assert fitness_values[2] == pytest.approx(0.4621, rel=1e-3)


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

    @pytest.fixture
    def mock_regime_detector(self):
        """Mock RegimeDetector"""
        detector = Mock(spec=RegimeDetector)
        detector.detect_regimes.return_value = np.array([0, 1, 2, 0, 1])
        return detector

    def test_standard_mode_initialization(
        self,
        mock_backtest_service,
        mock_strategy_factory,
        mock_gene_generator,
        mock_regime_detector,
    ):
        """標準GAモードでの初期化を確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
            regime_detector=mock_regime_detector,
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
        mock_regime_detector,
    ):
        """ハイブリッドGA+MLモードでの初期化を確認"""
        mock_predictor = Mock()
        mock_feature_adapter = Mock()

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
            regime_detector=mock_regime_detector,
            hybrid_mode=True,
            hybrid_predictor=mock_predictor,
            hybrid_feature_adapter=mock_feature_adapter,
        )

        # ハイブリッドモードであることを確認
        assert engine.hybrid_mode is True
        # HybridIndividualEvaluatorが使用されていることを確認
        assert type(engine.individual_evaluator).__name__ == "HybridIndividualEvaluator"

    def test_engine_without_regime_detector(
        self,
        mock_backtest_service,
        mock_strategy_factory,
        mock_gene_generator,
    ):
        """RegimeDetectorなしでの初期化を確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
            regime_detector=None,  # RegimeDetectorなし
            hybrid_mode=False,
            hybrid_predictor=None,
            hybrid_feature_adapter=None,
        )

        # 正常に初期化されることを確認
        assert engine.hybrid_mode is False
        assert isinstance(engine.individual_evaluator, IndividualEvaluator)

    def test_engine_components_are_set(
        self,
        mock_backtest_service,
        mock_strategy_factory,
        mock_gene_generator,
        mock_regime_detector,
    ):
        """エンジンのコンポーネントが正しく設定されることを確認"""
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
            regime_detector=mock_regime_detector,
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
