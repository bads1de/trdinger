"""
GAEngineFactoryのユニットテスト
"""

import pytest
from unittest.mock import Mock

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.core.ga_engine_factory import GeneticAlgorithmEngineFactory
from app.services.auto_strategy.config.ga import GAConfig
from app.services.backtest.backtest_service import BacktestService


class TestGAEngineFactory:
    """GAEngineFactoryのテストクラス"""

    def test_create_engine_basic(self):
        """標準的なエンジンの作成テスト"""
        mock_backtest_service = Mock(spec=BacktestService)
        ga_config = GAConfig()
        
        engine = GeneticAlgorithmEngineFactory.create_engine(
            backtest_service=mock_backtest_service,
            ga_config=ga_config
        )
        
        assert isinstance(engine, GeneticAlgorithmEngine)
        assert engine.individual_evaluator is not None
        assert engine.gene_generator is not None

    def test_create_engine_with_custom_config(self):
        """カスタム設定を用いたエンジンの作成テスト"""
        mock_backtest_service = Mock(spec=BacktestService)
        # 既存のフィールドを使用
        ga_config = GAConfig(
            population_size=10
        )
        
        engine = GeneticAlgorithmEngineFactory.create_engine(
            backtest_service=mock_backtest_service,
            ga_config=ga_config
        )
        
        assert isinstance(engine, GeneticAlgorithmEngine)
        # engine.gene_generator.config を確認
        assert engine.gene_generator.config.population_size == 10

    def test_singleton_behavior_not_enforced(self):
        """毎回新しいインスタンスが生成されることを確認（設計上の確認）"""
        mock_backtest_service = Mock(spec=BacktestService)
        ga_config = GAConfig()
        
        engine1 = GeneticAlgorithmEngineFactory.create_engine(mock_backtest_service, ga_config)
        engine2 = GeneticAlgorithmEngineFactory.create_engine(mock_backtest_service, ga_config)
        
        assert engine1 is not engine2
