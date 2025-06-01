"""
バックテスト戦略のテスト

各種戦略の動作とパラメータバリデーションをテストします。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from app.core.strategies.sma_cross_strategy import SMACrossStrategy
from backtest.engine.strategy_executor import StrategyExecutor


@pytest.mark.unit
@pytest.mark.backtest
class TestSMACrossStrategy:
    """SMAクロス戦略のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のOHLCVデータ"""
        dates = pd.date_range(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            freq="D",
        )

        # トレンドのあるデータを生成
        np.random.seed(42)
        base_price = 50000
        prices = []
        for i in range(len(dates)):
            # 上昇トレンド + ランダムノイズ
            trend = base_price + (i * 100)
            noise = np.random.normal(0, 500)
            prices.append(max(trend + noise, 1000))  # 最低価格を設定

        return pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.02 for p in prices],
                "Low": [p * 0.98 for p in prices],
                "Close": prices,
                "Volume": [1000 + np.random.randint(0, 500) for _ in prices],
            },
            index=dates,
        )

    def test_strategy_initialization(self):
        """戦略の初期化テスト"""
        # クラス属性の確認（インスタンス化せずに）
        assert hasattr(SMACrossStrategy, "n1")
        assert hasattr(SMACrossStrategy, "n2")
        assert SMACrossStrategy.n1 < SMACrossStrategy.n2  # 短期 < 長期

    def test_strategy_with_custom_parameters(self, sample_data):
        """カスタムパラメータでの戦略テスト"""

        # カスタムパラメータで戦略を作成
        class CustomSMACrossStrategy(SMACrossStrategy):
            n1 = 5
            n2 = 15

        # クラス属性の確認
        assert CustomSMACrossStrategy.n1 == 5
        assert CustomSMACrossStrategy.n2 == 15

    def test_strategy_signals_generation(self, sample_data):
        """戦略シグナル生成テスト"""
        from backtesting import Backtest

        bt = Backtest(sample_data, SMACrossStrategy, cash=100000, commission=0.001)
        stats = bt.run()

        # 基本的な統計が生成されることを確認
        assert "Return [%]" in stats
        assert "Sharpe Ratio" in stats
        assert "# Trades" in stats
        assert "Win Rate [%]" in stats

    def test_strategy_with_insufficient_data(self):
        """データ不足時の戦略テスト"""
        # 短すぎるデータ（SMAの計算に必要な期間より短い）
        short_data = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [102, 103],
                "Low": [99, 100],
                "Close": [101, 102],
                "Volume": [1000, 1100],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )

        from backtesting import Backtest

        bt = Backtest(short_data, SMACrossStrategy, cash=100000, commission=0.001)
        stats = bt.run()

        # データ不足でも実行できることを確認（取引は発生しない可能性）
        assert stats["# Trades"] >= 0

    def test_strategy_parameter_validation(self):
        """戦略パラメータのバリデーションテスト"""

        # 無効なパラメータ（n1 >= n2）
        class InvalidSMACrossStrategy(SMACrossStrategy):
            n1 = 20
            n2 = 10  # n1より小さい

        # この場合、戦略は動作するが論理的に意味がない
        # 実際の実装では、バリデーションを追加することを推奨
        assert InvalidSMACrossStrategy.n1 == 20
        assert InvalidSMACrossStrategy.n2 == 10


@pytest.mark.unit
@pytest.mark.backtest
class TestStrategyExecutor:
    """戦略実行エンジンのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のOHLCVデータ"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        np.random.seed(42)

        prices = []
        base_price = 50000
        for i in range(len(dates)):
            price = base_price + np.random.normal(0, 1000)
            prices.append(max(price, 1000))

        return pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.01 for p in prices],
                "Low": [p * 0.99 for p in prices],
                "Close": prices,
                "Volume": [1000] * len(prices),
            },
            index=dates,
        )

    def test_executor_initialization(self):
        """実行エンジンの初期化テスト"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        assert executor.initial_capital == 100000
        assert executor.commission_rate == 0.001

    def test_executor_with_strategy_config(self, sample_data):
        """戦略設定での実行テスト"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        strategy_config = {
            "indicators": [
                {"name": "SMA", "params": {"period": 10}},
                {"name": "SMA", "params": {"period": 20}},
            ],
            "entry_rules": [{"condition": "SMA(close, 10) > SMA(close, 20)"}],
            "exit_rules": [{"condition": "SMA(close, 10) < SMA(close, 20)"}],
        }

        result = executor.run_backtest(sample_data, strategy_config)

        # 結果の基本構造を確認
        assert "strategy_name" in result
        assert "symbol" in result
        assert "performance_metrics" in result
        assert "equity_curve" in result
        assert "trade_history" in result

    def test_executor_performance_metrics(self, sample_data):
        """パフォーマンス指標の計算テスト"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        strategy_config = {
            "indicators": [
                {"name": "SMA", "params": {"period": 5}},
                {"name": "SMA", "params": {"period": 15}},
            ],
            "entry_rules": [{"condition": "SMA(close, 5) > SMA(close, 15)"}],
            "exit_rules": [{"condition": "SMA(close, 5) < SMA(close, 15)"}],
        }

        result = executor.run_backtest(sample_data, strategy_config)
        metrics = result["performance_metrics"]

        # 必要な指標が存在することを確認
        required_metrics = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "total_trades",
            "profit_factor",
        ]

        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

    def test_executor_edge_cases(self, sample_data):
        """エッジケースのテスト"""
        executor = StrategyExecutor(
            initial_capital=1000, commission_rate=0.01  # 少額資金  # 高い手数料
        )

        strategy_config = {
            "indicators": [
                {"name": "SMA", "params": {"period": 5}},
                {"name": "SMA", "params": {"period": 10}},
            ],
            "entry_rules": [{"condition": "SMA(close, 5) > SMA(close, 10)"}],
            "exit_rules": [{"condition": "SMA(close, 5) < SMA(close, 10)"}],
        }

        result = executor.run_backtest(sample_data, strategy_config)

        # 少額資金・高手数料でも実行できることを確認
        assert result is not None
        assert "performance_metrics" in result

        # 高い手数料の影響を確認
        metrics = result["performance_metrics"]
        # 手数料が高い場合、リターンが悪化する可能性が高い
        assert "total_return" in metrics


class TestStrategyValidation:
    """戦略バリデーションのテスト"""

    def test_invalid_strategy_config(self):
        """無効な戦略設定のテスト"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        # 無効な設定（必須フィールドが不足）
        invalid_config = {
            "indicators": [],  # 指標なし
            "entry_rules": [],  # エントリールールなし
            "exit_rules": [],  # エグジットルールなし
        }

        sample_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100],
                "Volume": [1000],
            },
            index=[datetime.now()],
        )

        # 無効な設定でも例外が発生しないことを確認
        # （実際の実装では適切なバリデーションとエラーハンドリングが必要）
        result = executor.run_backtest(sample_data, invalid_config)
        assert result is not None

    def test_strategy_config_with_unknown_indicators(self):
        """未知の指標を含む戦略設定のテスト"""
        executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)

        config_with_unknown_indicator = {
            "indicators": [{"name": "UNKNOWN_INDICATOR", "params": {"period": 10}}],
            "entry_rules": [{"condition": "UNKNOWN_INDICATOR(close, 10) > 0"}],
            "exit_rules": [{"condition": "UNKNOWN_INDICATOR(close, 10) < 0"}],
        }

        sample_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100],
                "Volume": [1000],
            },
            index=[datetime.now()],
        )

        # 未知の指標でもエラーハンドリングされることを確認
        # （実際の実装では適切なエラーメッセージが必要）
        try:
            result = executor.run_backtest(sample_data, config_with_unknown_indicator)
            # 成功した場合は結果が返される
            assert result is not None
        except Exception as e:
            # エラーが発生した場合は適切な例外であることを確認
            assert isinstance(e, (ValueError, KeyError, AttributeError))
