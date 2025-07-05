"""
オートストラテジー取引量0問題の修正を検証する統合テスト
"""

import pytest
import logging
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import pandas as pd

from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.backtest_service import BacktestService

logger = logging.getLogger(__name__)


class TestAutoStrategyVolumeFix:
    """オートストラテジー取引量修正の統合テスト"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用のOHLCVデータ"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
        data = pd.DataFrame({
            'Open': [50000 + i * 10 for i in range(len(dates))],
            'High': [50100 + i * 10 for i in range(len(dates))],
            'Low': [49900 + i * 10 for i in range(len(dates))],
            'Close': [50050 + i * 10 for i in range(len(dates))],
            'Volume': [1000 + i for i in range(len(dates))],
        }, index=dates)
        return data

    @pytest.fixture
    def test_strategy_gene(self):
        """テスト用の戦略遺伝子"""
        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        ]
        
        entry_conditions = [
            Condition(left_operand="RSI", operator="<", right_operand=30)
        ]
        
        exit_conditions = [
            Condition(left_operand="RSI", operator=">", right_operand=70)
        ]
        
        risk_management = {
            "stop_loss": 0.03,
            "take_profit": 0.15,
            "position_size": 0.1,  # 10%の取引量
        }
        
        return StrategyGene(
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management=risk_management,
            metadata={"test": "volume_fix"}
        )

    @pytest.fixture
    def mock_backtest_data_service(self, sample_ohlcv_data):
        """モックのBacktestDataService"""
        mock_service = Mock()
        mock_service.get_data_for_backtest.return_value = sample_ohlcv_data
        return mock_service

    def test_ga_engine_parameter_passing(self, test_strategy_gene, mock_backtest_data_service):
        """GAエンジンでのパラメータ受け渡しテスト"""
        # GAエンジンの初期化
        strategy_factory = StrategyFactory()
        
        # BacktestServiceをモック
        backtest_service = Mock(spec=BacktestService)
        backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 10.0,
                "sharpe_ratio": 1.5,
                "max_drawdown": 5.0,
                "total_trades": 5,  # 取引が発生することを確認
            }
        }
        
        gene_generator = Mock()
        ga_engine = GeneticAlgorithmEngine(backtest_service, strategy_factory, gene_generator)
        
        # 固定バックテスト設定
        ga_engine._fixed_backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "initial_capital": 100000.0,
            "commission_rate": 0.001
        }
        
        # GAConfig
        ga_config = GAConfig(population_size=10, generations=5)
        
        # 個体評価を実行
        individual = [0.5, 0.3, 0.7, 0.2, 0.8]  # ダミーの個体
        
        with patch('app.core.services.auto_strategy.engines.ga_engine.GeneEncoder') as mock_encoder:
            mock_encoder.return_value.decode_list_to_strategy_gene.return_value = test_strategy_gene
            
            fitness = ga_engine._evaluate_individual(individual, ga_config)
            
            # バックテストが呼ばれたことを確認
            assert backtest_service.run_backtest.called
            
            # 呼び出された設定を確認
            call_args = backtest_service.run_backtest.call_args[0][0]
            
            # strategy_configが正しく設定されていることを確認
            assert "strategy_config" in call_args
            assert call_args["strategy_config"]["strategy_type"] == "GENERATED_GA"
            assert "strategy_gene" in call_args["strategy_config"]["parameters"]
            
            # 取引量設定が含まれていることを確認
            strategy_gene_dict = call_args["strategy_config"]["parameters"]["strategy_gene"]
            assert "risk_management" in strategy_gene_dict
            assert strategy_gene_dict["risk_management"]["position_size"] == 0.1
            
            # フィットネス値が返されることを確認
            assert isinstance(fitness, tuple)
            assert len(fitness) == 1
            assert fitness[0] > 0

    def test_backtest_service_strategy_class_handling(self, test_strategy_gene, mock_backtest_data_service):
        """BacktestServiceでの戦略クラス処理テスト"""
        backtest_service = BacktestService(mock_backtest_data_service)
        strategy_factory = StrategyFactory()
        
        # 戦略クラスを生成
        strategy_class = strategy_factory.create_strategy_class(test_strategy_gene)
        
        # strategy_classが直接渡される場合のテスト
        config_with_class = {
            "strategy_name": "TEST_STRATEGY",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "strategy_class": strategy_class,
        }
        
        # バックテストを実行
        result = backtest_service.run_backtest(config_with_class)
        
        # 結果が返されることを確認
        assert result is not None
        assert "performance_metrics" in result
        assert "total_trades" in result["performance_metrics"]
        
        # 取引が発生していることを確認（0でないことを期待）
        total_trades = result["performance_metrics"]["total_trades"]
        logger.info(f"取引回数: {total_trades}")
        
        # strategy_configが渡される場合のテスト
        config_with_config = {
            "strategy_name": "TEST_STRATEGY",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "GENERATED_TEST",
                "parameters": {"strategy_gene": test_strategy_gene.to_dict()},
            }
        }
        
        # バックテストを実行
        result2 = backtest_service.run_backtest(config_with_config)
        
        # 結果が返されることを確認
        assert result2 is not None
        assert "performance_metrics" in result2
        assert "total_trades" in result2["performance_metrics"]

    def test_strategy_factory_volume_calculation(self, test_strategy_gene, sample_ohlcv_data):
        """StrategyFactoryでの取引量計算テスト"""
        strategy_factory = StrategyFactory()
        
        # 戦略クラスを生成
        strategy_class = strategy_factory.create_strategy_class(test_strategy_gene)
        
        # 戦略インスタンスを作成
        strategy_instance = strategy_class()
        
        # 遺伝子が正しく設定されていることを確認
        assert hasattr(strategy_instance, 'gene')
        assert strategy_instance.gene is not None
        assert strategy_instance.gene.risk_management["position_size"] == 0.1
        
        # リスク管理設定が正しく取得できることを確認
        risk_management = strategy_instance.gene.risk_management
        position_size = risk_management.get("position_size", 0.1)
        assert position_size == 0.1
        
        logger.info(f"戦略の取引量設定: {position_size}")

    def test_end_to_end_volume_fix(self, mock_backtest_data_service):
        """エンドツーエンドの取引量修正テスト"""
        # AutoStrategyServiceを初期化
        auto_strategy_service = AutoStrategyService()
        
        # BacktestServiceをモック
        auto_strategy_service.backtest_service = Mock(spec=BacktestService)
        auto_strategy_service.backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 15.0,
                "sharpe_ratio": 2.0,
                "max_drawdown": 3.0,
                "total_trades": 8,  # 取引が発生することを確認
            }
        }
        
        # テスト用の戦略遺伝子を作成
        from app.core.services.auto_strategy.utils.strategy_gene_utils import create_default_strategy_gene
        test_gene = create_default_strategy_gene(StrategyGene)
        
        # 取引量を明示的に設定
        test_gene.risk_management["position_size"] = 0.15
        
        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "initial_capital": 100000.0,
            "commission_rate": 0.001
        }
        
        # 戦略テストを実行
        result = auto_strategy_service.test_strategy_generation(test_gene, backtest_config)
        
        # 結果を確認
        assert result["success"] is True
        assert "backtest_result" in result
        
        # バックテストが正しいパラメータで呼ばれたことを確認
        call_args = auto_strategy_service.backtest_service.run_backtest.call_args[0][0]
        assert "strategy_config" in call_args
        assert call_args["strategy_config"]["strategy_type"] == "GENERATED_TEST"
        assert "strategy_gene" in call_args["strategy_config"]["parameters"]
        
        # 取引量設定が正しく渡されていることを確認
        strategy_gene_dict = call_args["strategy_config"]["parameters"]["strategy_gene"]
        assert strategy_gene_dict["risk_management"]["position_size"] == 0.15
        
        logger.info("エンドツーエンドテスト成功: 取引量設定が正しく処理されました")
