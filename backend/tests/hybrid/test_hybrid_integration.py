"""
ハイブリッドGA統合テスト

エンドツーエンドのハイブリッドGA実行をテスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from app.services.auto_strategy.config.ga_runtime import GAConfig
from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.backtest.backtest_service import BacktestService


class TestHybridGAIntegration:
    """ハイブリッドGA統合テストクラス"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """サンプルOHLCVデータ"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        data = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(40000, 41000, 100),
            "high": np.random.uniform(41000, 42000, 100),
            "low": np.random.uniform(39000, 40000, 100),
            "close": np.random.uniform(40000, 41000, 100),
            "volume": np.random.uniform(100, 1000, 100),
        })
        return data

    @pytest.fixture
    def ga_config_hybrid(self):
        """ハイブリッドモード有効のGAConfig"""
        config = GAConfig(
            population_size=10,
            generations=2,
            crossover_rate=0.8,
            mutation_rate=0.2,
            hybrid_mode=True,
            hybrid_model_type="lightgbm",
            fitness_weights={
                "total_return": 0.3,
                "sharpe_ratio": 0.3,
                "max_drawdown": 0.2,
                "win_rate": 0.1,
                "prediction_score": 0.1,  # ML予測スコア
            },
            fallback_start_date="2023-01-01",
            fallback_end_date="2023-04-09",
        )
        return config

    @pytest.fixture
    def ga_config_standard(self):
        """標準GAモードのGAConfig"""
        config = GAConfig(
            population_size=10,
            generations=2,
            crossover_rate=0.8,
            mutation_rate=0.2,
            hybrid_mode=False,
            fallback_start_date="2023-01-01",
            fallback_end_date="2023-04-09",
        )
        return config

    def test_ga_engine_initialization_hybrid_mode(self, ga_config_hybrid):
        """
        ハイブリッドモードでのGAEngine初期化テスト
        
        検証項目:
        - hybrid_mode=Trueで初期化できる
        - HybridIndividualEvaluatorが使用される
        """
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        from app.services.auto_strategy.utils.hybrid_feature_adapter import HybridFeatureAdapter
        
        # Mock services
        mock_backtest_service = Mock(spec=BacktestService)
        strategy_factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config_hybrid)
        
        # HybridPredictor and FeatureAdapter
        hybrid_predictor = HybridPredictor(
            trainer_type="single",
            model_type="lightgbm",
        )
        hybrid_feature_adapter = HybridFeatureAdapter()
        
        # GAEngine初期化
        ga_engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=strategy_factory,
            gene_generator=gene_generator,
            hybrid_mode=True,
            hybrid_predictor=hybrid_predictor,
            hybrid_feature_adapter=hybrid_feature_adapter,
        )
        
        # 検証
        assert ga_engine.hybrid_mode is True
        assert ga_engine.individual_evaluator is not None
        # HybridIndividualEvaluatorが使用されていることを確認
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator
        )
        assert isinstance(ga_engine.individual_evaluator, HybridIndividualEvaluator)

    def test_ga_engine_initialization_standard_mode(self, ga_config_standard):
        """
        標準モードでのGAEngine初期化テスト
        
        検証項目:
        - hybrid_mode=Falseで初期化できる
        - IndividualEvaluatorが使用される
        """
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
        
        # Mock services
        mock_backtest_service = Mock(spec=BacktestService)
        strategy_factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config_standard)
        
        # GAEngine初期化
        ga_engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=strategy_factory,
            gene_generator=gene_generator,
            hybrid_mode=False,
        )
        
        # 検証
        assert ga_engine.hybrid_mode is False
        assert ga_engine.individual_evaluator is not None
        # IndividualEvaluatorが使用されていることを確認
        assert type(ga_engine.individual_evaluator).__name__ == "IndividualEvaluator"

    def test_experiment_manager_initialization_hybrid(self, ga_config_hybrid):
        """
        ExperimentManagerでのハイブリッドモード初期化テスト
        
        検証項目:
        - ExperimentManagerがハイブリッドモードでGAEngineを初期化できる
        """
        from app.services.auto_strategy.services.experiment_manager import ExperimentManager
        from app.services.auto_strategy.services.experiment_persistence_service import (
            ExperimentPersistenceService
        )
        
        # Mock services
        mock_backtest_service = Mock(spec=BacktestService)
        mock_persistence_service = Mock(spec=ExperimentPersistenceService)
        
        # ExperimentManager初期化
        experiment_manager = ExperimentManager(
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
        )
        
        # GAEngine初期化（ハイブリッドモード）
        experiment_manager.initialize_ga_engine(ga_config_hybrid)
        
        # 検証
        assert experiment_manager.ga_engine is not None
        assert experiment_manager.ga_engine.hybrid_mode is True

    def test_ga_config_hybrid_settings(self):
        """
        GAConfigのハイブリッド設定テスト
        
        検証項目:
        - hybrid_mode, hybrid_model_typeなどが正しく設定される
        """
        config = GAConfig(
            population_size=50,
            generations=10,
            hybrid_mode=True,
            hybrid_model_type="xgboost",
            hybrid_model_types=["lightgbm", "xgboost"],
            hybrid_automl_config={
                "enabled": True,
                "feature_generation": {
                    "tsfresh_enabled": True,
                }
            }
        )
        
        # 検証
        assert config.hybrid_mode is True
        assert config.hybrid_model_type == "xgboost"
        assert config.hybrid_model_types == ["lightgbm", "xgboost"]
        assert config.hybrid_automl_config["enabled"] is True

    @pytest.mark.skip(reason="実際のGA実行は時間がかかるためスキップ")
    def test_hybrid_ga_run_evolution(self, ga_config_hybrid, sample_ohlcv_data):
        """
        ハイブリッドGAの進化実行テスト（統合テスト）
        
        このテストは実際のGA進化を実行するため、通常はスキップします。
        必要に応じて手動で実行してください。
        """
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        from app.services.auto_strategy.utils.hybrid_feature_adapter import HybridFeatureAdapter
        
        # Mock backtest service
        mock_backtest_service = Mock(spec=BacktestService)
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 0.2,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 100,
            },
            "trade_history": [],
        }
        
        # Mock data service
        mock_data_service = Mock()
        mock_data_service.get_ohlcv_data.return_value = sample_ohlcv_data
        mock_backtest_service.data_service = mock_data_service
        mock_backtest_service._ensure_data_service_initialized = Mock()
        
        # Components
        strategy_factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config_hybrid)
        hybrid_predictor = HybridPredictor(
            trainer_type="single",
            model_type="lightgbm",
        )
        hybrid_feature_adapter = HybridFeatureAdapter()
        
        # GAEngine
        ga_engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=strategy_factory,
            gene_generator=gene_generator,
            hybrid_mode=True,
            hybrid_predictor=hybrid_predictor,
            hybrid_feature_adapter=hybrid_feature_adapter,
        )
        
        # バックテスト設定
        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-04-09",
        }
        
        # GA実行
        result = ga_engine.run_evolution(ga_config_hybrid, backtest_config)
        
        # 検証
        assert result is not None
        assert "best_individual" in result
        assert "population" in result
