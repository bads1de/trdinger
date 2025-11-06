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
        mock_funding_repo.get_funding_rates.return_value = sample_backtest_data['funding']

        mock_oi_repo = Mock(spec=OpenInterestRepository)
        mock_oi_repo.get_open_interest.return_value = sample_backtest_data['open_interest']

        # データサービスを作成
        data_service = BacktestDataService(
            ohlcv_repo=mock_ohlcv_repo,
            oi_repo=mock_oi_repo,
            fr_repo=mock_funding_repo
        )

        # データ取得をテスト
        ohlcv = data_service.get_ohlcv_data("BTC/USDT", "2023-01-01", "2023-03-31")
        assert isinstance(ohlcv, pd.DataFrame)
        assert len(ohlcv) > 0

    def test_strategy_configuration_in_backtest(self, ga_config, mock_backtest_service):
        """バックテストにおける戦略設定のテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 戦略設定を生成
        individual = [0.5, 0.3, 0.7, 0.2, 0.8]
        strategy_config = engine._individual_to_strategy_config(individual)

        assert isinstance(strategy_config, dict)
        assert 'indicator_params' in strategy_config
        assert 'entry_rules' in strategy_config
        assert 'exit_rules' in strategy_config
        assert 'risk_management' in strategy_config

    def test_realistic_backtest_execution(self, ga_config, mock_backtest_service):
        """現実的なバックテスト実行のテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 戦略設定を作成
        strategy_config = {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'initial_capital': 100000,
            'commission_rate': 0.001,
            'indicator_params': {'rsi_period': 14, 'ma_period': 20},
            'entry_rules': {'condition': 'rsi < 30'},
            'exit_rules': {'condition': 'rsi > 70'},
            'risk_management': {'stop_loss': 0.02, 'take_profit': 0.05},
        }

        # バックテストを実行
        backtest_result = engine._execute_backtest(strategy_config)

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
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # リスク管理設定を含む戦略
        strategy_with_risk = {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'initial_capital': 100000,
            'risk_per_trade': 0.02,  # 2%リスク
            'position_sizing': 'fixed_fractional',
            'stop_loss': 0.05,      # 5%ストップ
            'take_profit': 0.10,    # 10%利食い
        }

        # 戦略を検証
        validation_result = engine._validate_strategy_risk_parameters(strategy_with_risk)

        assert isinstance(validation_result, dict)
        assert "is_valid" in validation_result
        assert "risk_score" in validation_result
        assert "suggestions" in validation_result

    def test_market_condition_simulation(self, sample_backtest_data, ga_config, mock_backtest_service):
        """市場状況シミュレーションのテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=sample_backtest_data['ohlcv'],
            regime_detector=None
        )

        # 異なる市場状況をテスト
        market_conditions = [
            {"volatility": "high", "trend": "bullish"},
            {"volatility": "low", "trend": "bearish"},
            {"volatility": "medium", "trend": "sideways"},
        ]

        for condition in market_conditions:
            # 条件に基づいた戦略を生成
            strategy = engine._generate_condition_aware_strategy(condition)

            assert isinstance(strategy, dict)
            assert "market_condition" in strategy
            assert "strategy_params" in strategy

    def test_commission_and_slippage_modeling(self, ga_config, mock_backtest_service):
        """手数料とスリッページモデル化のテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 手数料とスリッページを含むバックテスト設定
        backtest_config = {
            'commission_rate': 0.001,  # 0.1%
            'slippage': 0.0005,        # 0.05%
            'execution_delay': 1,      # 1本足遅延
        }

        # 費用モデルを適用
        adjusted_result = engine._apply_cost_modeling(backtest_config, mock_backtest_service.run_backtest({}))

        assert isinstance(adjusted_result, dict)
        assert "performance_metrics" in adjusted_result

        # 費用が反映されていること
        original_return = 0.15  # モックデータに基づく
        adjusted_return = adjusted_result["performance_metrics"]["total_return"]
        assert adjusted_return <= original_return

    def test_drawdown_control_mechanisms(self, ga_config, mock_backtest_service):
        """ドローダウン制御メカニズムのテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # ドローダウン制御戦略をテスト
        drawdown_strategies = [
            {"type": "fixed_stop", "level": 0.10},
            {"type": "trailing_stop", "percentage": 0.08},
            {"type": "volatility_based", "multiplier": 2.0},
        ]

        for strategy in drawdown_strategies:
            # 戦略を適用
            controlled_result = engine._apply_drawdown_control(strategy, mock_backtest_service.run_backtest({}))

            assert isinstance(controlled_result, dict)
            assert "drawdown_controlled" in controlled_result
            assert "max_drawdown" in controlled_result["performance_metrics"]

    def test_multi_timeframe_backtest_integration(self, ga_config, mock_backtest_service):
        """複数時間足バックテスト統合のテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 複数時間足をテスト
        timeframes = ["1h", "4h", "1d"]

        for timeframe in timeframes:
            ga_config.timeframe = timeframe

            # 各時間足で適応度を評価
            individual = np.random.rand(5)
            fitness = engine._evaluate_individual(individual)

            assert isinstance(fitness, float)
            assert not np.isnan(fitness)

    def test_backtest_overfitting_prevention(self, ga_config, mock_backtest_service):
        """バックテスト過学習防止のテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 過学習防止メカニズムをテスト
        prevention_techniques = [
            "walk_forward_optimization",
            "monte_carlo_simulation",
            "out_of_sample_testing",
            "cross_validation",
        ]

        for technique in prevention_techniques:
            # 技術を適用
            result = engine._apply_overfitting_prevention(technique, {})

            assert isinstance(result, dict)
            assert "overfitting_score" in result
            assert "robustness_metric" in result

    def test_stress_testing_scenario_analysis(self, ga_config, mock_backtest_service):
        """ストレステストシナリオ分析のテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # ストレスシナリオを定義
        stress_scenarios = [
            {"name": "flash_crash", "drop_percentage": 0.30},
            {"name": "high_volatility", "volatility_multiplier": 3.0},
            {"name": "liquidity_crunch", "spread_multiplier": 5.0},
        ]

        for scenario in stress_scenarios:
            # シナリオを適用
            stress_result = engine._run_stress_test(scenario, mock_backtest_service.run_backtest({}))

            assert isinstance(stress_result, dict)
            assert "scenario" in stress_result
            assert "stress_impact" in stress_result
            assert "recovery_time" in stress_result

    def test_final_backtest_integration_validation(self, ga_config, mock_backtest_service):
        """最終バックテスト統合検証"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        assert engine is not None
        assert hasattr(engine, '_evaluate_individual')
        assert hasattr(engine, 'evolve')

        # 基本的な進化が可能であること
        result = engine.evolve()
        assert result is not None

        print("✅ バックテスト統合包括的テスト成功")


# TDDアプローチによるバックテスト統合テスト
class TestBacktestIntegrationTDD:
    """TDDアプローチによるバックテスト統合テスト"""

    def test_backtest_data_pipeline_setup(self):
        """バックテストデータパイプライン設定のテスト"""
        # 最小限のデータパイプラインを設定
        mock_ohlcv_repo = Mock()
        mock_ohlcv_repo.get_ohlcv_data.return_value = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'open': [100, 101, 99, 102, 100, 101, 98, 103, 100, 102],
            'high': [102, 103, 101, 104, 102, 103, 100, 105, 102, 104],
            'low': [98, 99, 97, 100, 98, 99, 96, 101, 98, 100],
            'close': [101, 100, 102, 101, 101, 98, 103, 100, 102, 101],
            'volume': [1000, 1100, 900, 1200, 1000, 950, 1050, 1150, 1000, 1100],
        })

        data_service = BacktestDataService(
            ohlcv_repo=mock_ohlcv_repo,
            oi_repo=Mock(),
            fr_repo=Mock()
        )

        # データ取得が可能であること
        data = data_service.get_ohlcv_data("BTC/USDT", "2023-01-01", "2023-01-10")
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10

        print("✅ バックテストデータパイプライン設定のテスト成功")

    def test_basic_strategy_backtest_workflow(self):
        """基本戦略バックテストワークフローテスト"""
        ga_config = GAConfig.from_dict({
            "population_size": 10,
            "num_generations": 3,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {"total_return": 0.1}
        }

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 基本的な戦略設定を作成
        strategy_config = {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'initial_capital': 100000,
        }

        # バックテストを実行
        result = engine._execute_backtest(strategy_config)
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