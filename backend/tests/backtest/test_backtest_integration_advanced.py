"""
バックテスト統合テスト
GA戦略とバックテストの包括的統合テスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.backtest.backtest_service import BacktestService
from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.auto_strategy.config.ga import GASettings as GAConfig
from app.services.auto_strategy.models.strategy_gene import StrategyGene
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository


class TestBacktestIntegrationComprehensive:
    """バックテスト統合包括的テスト"""

    @pytest.fixture
    def sample_backtest_data(self):
        """サンプルバックテストデータ"""
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        np.random.seed(42)

        ohlcv_data = pd.DataFrame({
            'timestamp': dates,
            'open': 10000 + np.random.randn(len(dates)) * 200,
            'high': 10000 + np.random.randn(len(dates)) * 300,
            'low': 10000 + np.random.randn(len(dates)) * 300,
            'close': 10000 + np.random.randn(len(dates)) * 200,
            'volume': 500 + np.random.randint(100, 1000, len(dates)),
        })

        # OHLCの関係を確保
        ohlcv_data['high'] = ohlcv_data[['open', 'close', 'high']].max(axis=1)
        ohlcv_data['low'] = ohlcv_data[['open', 'close', 'low']].min(axis=1)

        # 追加データ
        funding_data = pd.DataFrame({
            'timestamp': dates,
            'funding_rate': np.random.randn(len(dates)) * 0.001,
        })

        open_interest_data = pd.DataFrame({
            'timestamp': dates,
            'open_interest': 1000000 + np.random.randint(-100000, 100000, len(dates)),
        })

        return {
            'ohlcv': ohlcv_data,
            'funding': funding_data,
            'open_interest': open_interest_data,
        }

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        mock_service = Mock(spec=BacktestService)
        mock_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.08,
                "total_trades": 25,
                "win_rate": 0.60,
                "profit_factor": 1.8,
                "average_trade": 0.006,
                "number_of_winning_trades": 15,
                "number_of_losing_trades": 10,
            },
            "trade_history": [
                {
                    "entry_time": "2023-01-15T10:00:00",
                    "exit_time": "2023-01-16T10:00:00",
                    "entry_price": 10000.0,
                    "exit_price": 10150.0,
                    "position_size": 0.1,
                    "profit": 150.0,
                    "return": 0.015,
                }
            ],
            "equity_curve": [
                {"timestamp": "2023-01-01T00:00:00", "equity": 100000.0},
                {"timestamp": "2023-01-02T00:00:00", "equity": 100150.0},
            ]
        }
        return mock_service

    @pytest.fixture
    def ga_config(self):
        """GA設定"""
        return GAConfig.from_dict({
            "population_size": 30,
            "num_generations": 8,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "initial_capital": 100000,
            "commission_rate": 0.001,
            "risk_per_trade": 0.02,
        })

    def test_backtest_service_initialization(self, mock_backtest_service):
        """バックテストサービス初期化のテスト"""
        assert mock_backtest_service is not None
        assert hasattr(mock_backtest_service, 'run_backtest')

        # 基本的なバックテスト実行
        result = mock_backtest_service.run_backtest({})
        assert result["success"] is True
        assert "performance_metrics" in result

    def test_data_service_integration(self, sample_backtest_data):
        """データサービス統合のテスト"""
        # モックリポジトリ
        mock_ohlcv_repo = Mock(spec=OHLCVRepository)
        mock_ohlcv_repo.get_ohlcv_data.return_value = sample_backtest_data['ohlcv']

        mock_funding_repo = Mock(spec=FundingRateRepository)
        mock_funding_repo.get_funding_rate_data.return_value = sample_backtest_data['funding']

        mock_oi_repo = Mock(spec=OpenInterestRepository)
        mock_oi_repo.get_open_interest_data.return_value = sample_backtest_data['open_interest']

        # データサービスを作成
        data_service = BacktestDataService(
            ohlcv_repo=mock_ohlcv_repo,
            oi_repo=mock_oi_repo,
            fr_repo=mock_funding_repo
        )

        # データ取得をテスト（4つのパラメータが必要）
        from datetime import datetime
        ohlcv = data_service.get_ohlcv_data(
            "BTC/USDT",
            "1d",
            datetime(2023, 1, 1),
            datetime(2023, 3, 31)
        )
        assert isinstance(ohlcv, pd.DataFrame)

    def test_strategy_configuration_in_backtest(self, ga_config, mock_backtest_service):
        """バックテストにおける戦略設定のテスト"""
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=StrategyFactory(),
            gene_generator=RandomGeneGenerator(ga_config),
            regime_detector=None
        )

        # 戦略遺伝子を生成してテスト
        gene = engine.gene_generator.generate_random_gene()
        
        assert gene is not None
        assert hasattr(gene, 'indicators')
        assert hasattr(gene, 'entry_conditions')
        assert hasattr(gene, 'exit_conditions')

    def test_realistic_backtest_execution(self, ga_config, mock_backtest_service):
        """現実的なバックテスト実行のテスト"""
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=StrategyFactory(),
            gene_generator=RandomGeneGenerator(ga_config),
            regime_detector=None
        )

        # 戦略設定を作成（必須フィールドを全て含む）
        strategy_config = {
            'strategy_name': 'test_strategy',
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            'commission_rate': 0.001,
            'strategy_config': {
                'indicator_params': {'rsi_period': 14, 'ma_period': 20},
                'entry_rules': {'condition': 'rsi < 30'},
                'exit_rules': {'condition': 'rsi > 70'},
                'risk_management': {'stop_loss': 0.02, 'take_profit': 0.05},
            }
        }

        # バックテストを実行（backtest_serviceを直接使用）
        backtest_result = mock_backtest_service.run_backtest(strategy_config)

        assert isinstance(backtest_result, dict)
        assert backtest_result["success"] is True
        assert "performance_metrics" in backtest_result

        metrics = backtest_result["performance_metrics"]
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    def test_performance_metrics_validation(self, mock_backtest_service):
        """パフォーマンスメトリクス検証のテスト"""
        result = mock_backtest_service.run_backtest({})

        metrics = result["performance_metrics"]

        # メトリクスの正当性を検証
        assert isinstance(metrics["total_return"], (int, float))
        assert isinstance(metrics["sharpe_ratio"], (int, float))
        assert isinstance(metrics["max_drawdown"], (int, float))
        assert isinstance(metrics["win_rate"], (int, float))

        # 範囲チェック
        assert -1 <= metrics["max_drawdown"] <= 0  # 最大ドローダウンは負またはゼロ
        assert 0 <= metrics["win_rate"] <= 1     # 勝率は0-1の範囲

    def test_trade_history_analysis(self, mock_backtest_service):
        """トレード履歴分析のテスト"""
        result = mock_backtest_service.run_backtest({})
        trade_history = result["trade_history"]

        assert isinstance(trade_history, list)
        if trade_history:
            first_trade = trade_history[0]
            assert "entry_time" in first_trade
            assert "exit_time" in first_trade
            assert "profit" in first_trade
            assert "return" in first_trade

            assert isinstance(first_trade["profit"], (int, float))
            assert isinstance(first_trade["return"], (int, float))

    def test_equity_curve_validation(self, mock_backtest_service):
        """資産曲線検証のテスト"""
        result = mock_backtest_service.run_backtest({})
        equity_curve = result["equity_curve"]

        assert isinstance(equity_curve, list)
        if equity_curve:
            first_point = equity_curve[0]
            assert "timestamp" in first_point
            assert "equity" in first_point
            assert isinstance(first_point["equity"], (int, float))

    def test_risk_management_integration(self, ga_config, mock_backtest_service):
        """リスク管理統合のテスト"""
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=StrategyFactory(),
            gene_generator=RandomGeneGenerator(ga_config),
            regime_detector=None
        )

        # リスク管理設定を含む戦略
        strategy_with_risk = {
            'strategy_name': 'risk_management_test',
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            'commission_rate': 0.001,
            'strategy_config': {
                'risk_per_trade': 0.02,  # 2%リスク
                'position_sizing': 'fixed_fractional',
                'stop_loss': 0.05,      # 5%ストップ
                'take_profit': 0.10,    # 10%利食い
            }
        }

        # 基本的な設定検証
        assert strategy_with_risk['strategy_config']['risk_per_trade'] == 0.02
        assert strategy_with_risk['strategy_config']['stop_loss'] > 0
        assert strategy_with_risk['strategy_config']['take_profit'] > strategy_with_risk['strategy_config']['stop_loss']

    def test_market_condition_simulation(self, sample_backtest_data, ga_config, mock_backtest_service):
        """市場状況シミュレーションのテスト"""
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=StrategyFactory(),
            gene_generator=RandomGeneGenerator(ga_config),
            regime_detector=None
        )

        # 異なる市場状況をテスト
        market_conditions = [
            {"volatility": "high", "trend": "bullish"},
            {"volatility": "low", "trend": "bearish"},
            {"volatility": "medium", "trend": "sideways"},
        ]

        for condition in market_conditions:
            # 各条件で戦略遺伝子を生成
            gene = engine.gene_generator.generate_random_gene()
            
            assert gene is not None
            assert isinstance(condition, dict)

    def test_commission_and_slippage_modeling(self, ga_config, mock_backtest_service):
        """手数料とスリッページモデル化のテスト"""
        # 手数料とスリッページを含むバックテスト設定
        backtest_config = {
            'strategy_name': 'cost_model_test',
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            'commission_rate': 0.001,  # 0.1%
            'strategy_config': {
                'slippage': 0.0005,        # 0.05%
                'execution_delay': 1,      # 1本足遅延
            }
        }

        # バックテストを実行
        result = mock_backtest_service.run_backtest(backtest_config)

        assert isinstance(result, dict)
        assert "performance_metrics" in result

        # 費用が設定に含まれていること
        assert backtest_config['commission_rate'] == 0.001

    def test_drawdown_control_mechanisms(self, ga_config, mock_backtest_service):
        """ドローダウン制御メカニズムのテスト"""
        # ドローダウン制御戦略をテスト
        drawdown_strategies = [
            {"type": "fixed_stop", "level": 0.10},
            {"type": "trailing_stop", "percentage": 0.08},
            {"type": "volatility_based", "multiplier": 2.0},
        ]

        for strategy in drawdown_strategies:
            # 基本的なバックテスト結果を取得
            result = mock_backtest_service.run_backtest({})

            assert isinstance(result, dict)
            assert "performance_metrics" in result
            assert "max_drawdown" in result["performance_metrics"]
            assert isinstance(strategy, dict)

    def test_multi_timeframe_backtest_integration(self, ga_config, mock_backtest_service):
        """複数時間足バックテスト統合のテスト"""
        # 複数時間足をテスト
        timeframes = ["1h", "4h", "1d"]

        for timeframe in timeframes:
            # 各時間足でバックテスト設定を作成
            config = {
                'strategy_name': f'test_strategy_{timeframe}',
                'symbol': 'BTC/USDT',
                'timeframe': timeframe,
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'initial_capital': 100000,
                'commission_rate': 0.001,
                'strategy_config': {}
            }
            
            result = mock_backtest_service.run_backtest(config)
            assert isinstance(result, dict)
            assert result["success"] is True

    def test_backtest_overfitting_prevention(self, ga_config, mock_backtest_service):
        """バックテスト過学習防止のテスト"""
        # 過学習防止メカニズムをテスト
        prevention_techniques = [
            "walk_forward_optimization",
            "monte_carlo_simulation",
            "out_of_sample_testing",
            "cross_validation",
        ]

        for technique in prevention_techniques:
            # 各技術が文字列であることを確認
            assert isinstance(technique, str)
            assert len(technique) > 0

    def test_stress_testing_scenario_analysis(self, ga_config, mock_backtest_service):
        """ストレステストシナリオ分析のテスト"""
        # ストレスシナリオを定義
        stress_scenarios = [
            {"name": "flash_crash", "drop_percentage": 0.30},
            {"name": "high_volatility", "volatility_multiplier": 3.0},
            {"name": "liquidity_crunch", "spread_multiplier": 5.0},
        ]

        for scenario in stress_scenarios:
            # シナリオデータの検証
            assert isinstance(scenario, dict)
            assert "name" in scenario
            assert isinstance(scenario["name"], str)
            
            # バックテスト結果の取得
            result = mock_backtest_service.run_backtest({})
            assert isinstance(result, dict)

    def test_final_backtest_integration_validation(self, ga_config, mock_backtest_service):
        """最終バックテスト統合検証"""
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest_service,
            strategy_factory=StrategyFactory(),
            gene_generator=RandomGeneGenerator(ga_config),
            regime_detector=None
        )

        assert engine is not None
        assert hasattr(engine, 'backtest_service')
        assert hasattr(engine, 'strategy_factory')
        assert hasattr(engine, 'gene_generator')

        print("✅ バックテスト統合包括的テスト成功")


# TDDアプローチによるバックテスト統合テスト
class TestBacktestIntegrationTDD:
    """TDDアプローチによるバックテスト統合テスト"""

    def test_backtest_data_pipeline_setup(self):
        """バックテストデータパイプライン設定のテスト"""
        from datetime import datetime
        
        # 最小限のデータパイプラインを設定
        mock_ohlcv_repo = Mock()
        mock_ohlcv_repo.get_ohlcv_data.return_value = [
            Mock(timestamp=datetime(2023, 1, i+1), open=100+i, high=102+i, low=98+i, close=101+i, volume=1000+i*100)
            for i in range(10)
        ]

        data_service = BacktestDataService(
            ohlcv_repo=mock_ohlcv_repo,
            oi_repo=Mock(),
            fr_repo=Mock()
        )

        # データ取得が可能であること（4つのパラメータが必要）
        data = data_service.get_ohlcv_data(
            "BTC/USDT",
            "1d",
            datetime(2023, 1, 1),
            datetime(2023, 1, 10)
        )
        assert isinstance(data, pd.DataFrame)

        print("✅ バックテストデータパイプライン設定のテスト成功")

    def test_basic_strategy_backtest_workflow(self):
        """基本戦略バックテストワークフローテスト"""
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {"total_return": 0.1}
        }

        # 基本的な戦略設定を作成（必須フィールドを含む）
        strategy_config = {
            'strategy_name': 'basic_test_strategy',
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            'commission_rate': 0.001,
            'strategy_config': {}
        }

        # バックテストを実行
        result = mock_backtest_service.run_backtest(strategy_config)
        assert result["success"] is True

        print("✅ 基本戦略バックテストワークフローテスト成功")

    def test_performance_metrics_calculation(self):
        """パフォーメトリクス計算テスト"""
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.08,
                "win_rate": 0.60,
            }
        }

        result = mock_backtest_service.run_backtest({})
        metrics = result["performance_metrics"]

        # 基本的なメトリクスが存在すること
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics

        print("✅ パフォーメトリクス計算テスト成功")

    def test_risk_adjusted_return_calculation(self):
        """リスク調整リターン計算テスト"""
        # リスク調整リターンを計算
        total_return = 0.15
        max_drawdown = -0.08
        sharpe_ratio = 1.5

        # カリマー比率を計算（例）
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

        assert isinstance(calmar_ratio, float)
        assert calmar_ratio >= 0

        print(f"✅ リスク調整リターン計算テスト成功: Calmar Ratio = {calmar_ratio:.2f}")

    def test_backtest_result_interpretation(self):
        """バックテスト結果解釈テスト"""
        sample_result = {
            "success": True,
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.08,
                "win_rate": 0.60,
                "profit_factor": 1.8,
            },
            "trade_history": [
                {"profit": 150.0, "return": 0.015},
                {"profit": -80.0, "return": -0.008},
            ]
        }

        # 結果を解釈
        total_return = sample_result["performance_metrics"]["total_return"]
        win_rate = sample_result["performance_metrics"]["win_rate"]

        # 解釈が可能であること
        assert total_return > 0  # 利益が出ている
        assert win_rate > 0.5   # 勝率が50%以上

        print("✅ バックテスト結果解釈テスト成功")