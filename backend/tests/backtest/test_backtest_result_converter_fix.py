"""
BacktestResultConverterの修正をテストするモジュール

performance_metricsフィールドが正しく作成されることを確認する
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from typing import Any, Dict

from app.services.backtest.conversion.backtest_result_converter import BacktestResultConverter


class TestBacktestResultConverterFix:
    """BacktestResultConverterの修正をテスト"""

    @pytest.fixture
    def converter(self):
        """テスト用のconverterインスタンス"""
        return BacktestResultConverter()

    @pytest.fixture
    def mock_stats_with_trades(self):
        """取引ありのモック統計データ - _Statsオブジェクト形式で模倣"""
        mock_stats = Mock()

        # _Statsオブジェクトの属性を模倣（辞書のように振る舞う）
        stats_data = {
            'Return [%]': 15.5,
            '# Trades': 25,
            'Win Rate [%]': 60.0,
            'Best Trade [%]': 5.2,
            'Worst Trade [%]': -3.1,
            'Avg. Trade [%]': 0.62,
            'Max. Drawdown [%]': 8.5,
            'Avg. Drawdown [%]': 3.2,
            'Max. Drawdown Duration': 15,
            'Avg. Drawdown Duration': 8.5,
            'Sharpe Ratio': 1.25,
            'Sortino Ratio': 0.95,
            'Calmar Ratio': 1.82,
            'Equity Final [$]': 115500.0,
            'Equity Peak [$]': 120000.0,
            'Buy & Hold Return [%]': 8.2,
            'Profit Factor': 1.35
        }

        mock_stats.keys = lambda: stats_data.keys()
        mock_stats.get = lambda key, default=0.0: stats_data.get(key, default)
        mock_stats.__call__ = lambda: mock_stats

        # Series形式の属性を持たないようにする（_Statsオブジェクトとして処理されるように）
        mock_stats.index = None
        mock_stats.values = None

        # 取引データ - 実際のDataFrameを模倣したクラスを使用
        class MockTradesDF:
            def __init__(self):
                self.data = [{'PnL': 1.0}]  # 勝ち取引
                self.columns = ['PnL']
                self.empty = False

            def __len__(self):
                return len(self.data)

            def iterrows(self):
                for i, row in enumerate(self.data):
                    yield i, row

        mock_trades_df = MockTradesDF()
        mock_stats._trades = mock_trades_df

        # エクイティカーブ
        class MockEquityDF:
            def __init__(self):
                self.data = [
                    (datetime(2020, 1, 1), {'Equity': 100000.0}),
                    (datetime(2020, 1, 2), {'Equity': 115500.0})
                ]
                self.columns = ['Equity']
                self.empty = False

            def __len__(self):
                return len(self.data)

            def iterrows(self):
                for timestamp, row in self.data:
                    yield timestamp, row

            @property
            def iloc(self):
                class ILocIndexer:
                    def __getitem__(self, idx):
                        if idx == 0:
                            return self.data[0][1]['Equity']
                        elif idx == -1:
                            return self.data[-1][1]['Equity']
                        return 0.0
                return ILocIndexer()

        mock_equity_df = MockEquityDF()
        mock_stats._equity_curve = mock_equity_df

        return mock_stats

    @pytest.fixture
    def mock_stats_no_trades(self):
        """取引なしのモック統計データ - _Statsオブジェクト形式で模倣"""
        mock_stats = Mock()

        # _Statsオブジェクトの属性を模倣（辞書のように振る舞う）
        stats_data = {
            'Return [%]': 0.0,
            '# Trades': 0,
            'Win Rate [%]': 0.0,
            'Best Trade [%]': 0.0,
            'Worst Trade [%]': 0.0,
            'Avg. Trade [%]': 0.0,
            'Max. Drawdown [%]': 0.0,
            'Avg. Drawdown [%]': 0.0,
            'Max. Drawdown Duration': 0,
            'Avg. Drawdown Duration': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Calmar Ratio': 0.0,
            'Equity Final [$]': 100000.0,
            'Equity Peak [$]': 100000.0,
            'Buy & Hold Return [%]': 0.0,
            'Profit Factor': 0.0
        }

        mock_stats.keys = lambda: stats_data.keys()
        mock_stats.get = lambda key, default=0.0: stats_data.get(key, default)
        mock_stats.__call__ = lambda: mock_stats

        # Series形式の属性を持たないようにする（_Statsオブジェクトとして処理されるように）
        mock_stats.index = None
        mock_stats.values = None

        # 空の取引データ - 実際のDataFrameを模倣したクラスを使用
        class MockEmptyTradesDF:
            def __init__(self):
                self.data = []
                self.columns = ['PnL']
                self.empty = True

            def __len__(self):
                return len(self.data)

            def iterrows(self):
                return iter([])

        mock_trades_df = MockEmptyTradesDF()
        mock_stats._trades = mock_trades_df

        # エクイティカーブ
        class MockEquityDF:
            def __init__(self):
                self.data = [
                    (datetime(2020, 1, 1), {'Equity': 100000.0}),
                    (datetime(2020, 1, 2), {'Equity': 100000.0})
                ]
                self.columns = ['Equity']
                self.empty = False

            def __len__(self):
                return len(self.data)

            def iterrows(self):
                for timestamp, row in self.data:
                    yield timestamp, row

            @property
            def iloc(self):
                class ILocIndexer:
                    def __getitem__(self, idx):
                        return 100000.0
                return ILocIndexer()

        mock_equity_df = MockEquityDF()
        mock_stats._equity_curve = mock_equity_df

        return mock_stats

    def test_performance_metrics_field_created(self, converter, mock_stats_with_trades):
        """performance_metricsフィールドが作成されることをテスト"""
        config_json = {"commission_rate": 0.001}

        result = converter.convert_backtest_results(
            stats=mock_stats_with_trades,
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=100000.0,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            config_json=config_json
        )

        # performance_metricsフィールドが存在することを確認
        assert "performance_metrics" in result
        assert isinstance(result["performance_metrics"], dict)

    def test_statistics_properly_mapped_to_performance_metrics(self, converter, mock_stats_with_trades):
        """統計情報がperformance_metricsに正しくマップされることをテスト"""
        config_json = {"commission_rate": 0.001}

        result = converter.convert_backtest_results(
            stats=mock_stats_with_trades,
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=100000.0,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            config_json=config_json
        )

        performance_metrics = result["performance_metrics"]

        # 主要な統計情報が含まれていることを確認
        expected_metrics = [
            "total_return", "total_trades", "win_rate", "best_trade", "worst_trade",
            "avg_trade", "max_drawdown", "avg_drawdown", "max_drawdown_duration",
            "avg_drawdown_duration", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "final_equity", "equity_peak", "buy_hold_return", "profit_factor"
        ]

        for metric in expected_metrics:
            assert metric in performance_metrics, f"{metric}がperformance_metricsに含まれていません"

    def test_performance_metrics_values_correct(self, converter, mock_stats_with_trades):
        """performance_metricsの値が正しいことをテスト"""
        config_json = {"commission_rate": 0.001}

        result = converter.convert_backtest_results(
            stats=mock_stats_with_trades,
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=100000.0,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            config_json=config_json
        )

        performance_metrics = result["performance_metrics"]

        # 値の検証
        assert performance_metrics["total_return"] == 15.5
        assert performance_metrics["total_trades"] == 25
        assert performance_metrics["win_rate"] == 60.0
        assert performance_metrics["sharpe_ratio"] == 1.25
        assert performance_metrics["final_equity"] == 115500.0

    def test_no_trades_scenario(self, converter, mock_stats_no_trades):
        """取引なしのシナリオでperformance_metricsが作成されることをテスト"""
        config_json = {"commission_rate": 0.001}

        result = converter.convert_backtest_results(
            stats=mock_stats_no_trades,
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=100000.0,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            config_json=config_json
        )

        performance_metrics = result["performance_metrics"]

        assert performance_metrics["total_trades"] == 0
        assert performance_metrics["total_return"] == 0.0
        assert performance_metrics["final_equity"] == 100000.0

    def test_experiment_persistence_service_integration(self, converter, mock_stats_with_trades):
        """experiment_persistence_serviceでの使用をシミュレート"""
        config_json = {"commission_rate": 0.001}

        # converterで結果を変換
        detailed_result = converter.convert_backtest_results(
            stats=mock_stats_with_trades,
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=100000.0,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            config_json=config_json
        )

        # experiment_persistence_serviceの_prepare_backtest_result_dataをシミュレート
        config = {
            "strategy_name": "test_strategy",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": datetime(2020, 1, 1),
            "end_date": datetime(2020, 12, 31),
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "strategy_config": {},
            "experiment_id": "test_exp_id",
            "db_experiment_id": 1,
            "fitness_score": 0.85
        }

        backtest_result_data = {
            "strategy_name": config["strategy_name"],
            "symbol": config["symbol"],
            "timeframe": config["timeframe"],
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "initial_capital": config["initial_capital"],
            "commission_rate": config.get("commission_rate", 0.001),
            "config_json": {
                "strategy_config": config["strategy_config"],
                "experiment_id": config["experiment_id"],
                "db_experiment_id": config["db_experiment_id"],
                "fitness_score": config["fitness_score"],
            },
            "performance_metrics": detailed_result.get("performance_metrics", {}),
            "equity_curve": detailed_result.get("equity_curve", []),
            "trade_history": detailed_result.get("trade_history", []),
            "execution_time": detailed_result.get("execution_time"),
            "status": "completed",
        }

        # performance_metricsが空でないことを確認
        assert backtest_result_data["performance_metrics"] != {}
        assert "total_return" in backtest_result_data["performance_metrics"]
        assert "total_trades" in backtest_result_data["performance_metrics"]
        assert backtest_result_data["performance_metrics"]["total_return"] == 15.5
        assert backtest_result_data["performance_metrics"]["total_trades"] == 25

    def test_realistic_trading_scenario(self, converter):
        """現実的な取引シナリオでのテスト（勝ちトレード3つ、負けトレード2つ）"""
        config_json = {"commission_rate": 0.001}

        # 現実的な統計データ
        mock_stats = Mock()

        # _Statsオブジェクト形式
        stats_data = {
            'Return [%]': 0.0,  # エクイティカーブから計算される
            '# Trades': 0,      # 取引データから計算される
            'Win Rate [%]': 0.0, # 取引データから計算される
            'Best Trade [%]': 0.0,
            'Worst Trade [%]': 0.0,
            'Avg. Trade [%]': 0.0,
            'Max. Drawdown [%]': 5.0,
            'Avg. Drawdown [%]': 2.0,
            'Max. Drawdown Duration': 10,
            'Avg. Drawdown Duration': 5.0,
            'Sharpe Ratio': 1.2,
            'Sortino Ratio': 0.8,
            'Calmar Ratio': 1.5,
            'Equity Final [$]': 100000.0,
            'Equity Peak [$]': 102000.0,
            'Buy & Hold Return [%]': 3.0,
            'Profit Factor': 0.0  # 取引データから計算される
        }

        mock_stats.keys = lambda: stats_data.keys()
        mock_stats.get = lambda key, default=0.0: stats_data.get(key, default)
        mock_stats.__call__ = lambda: mock_stats
        mock_stats.index = None
        mock_stats.values = None

        # 現実的な取引データ（勝ち3つ、負け2つ）
        trades_data = [
            {'EntryTime': datetime(2020, 1, 1), 'ExitTime': datetime(2020, 1, 2), 'EntryPrice': 100.0, 'ExitPrice': 102.0, 'Size': 1.0, 'PnL': 2.0, 'ReturnPct': 2.0},  # 勝ち
            {'EntryTime': datetime(2020, 1, 3), 'ExitTime': datetime(2020, 1, 4), 'EntryPrice': 102.0, 'ExitPrice': 101.0, 'Size': 1.0, 'PnL': -1.0, 'ReturnPct': -1.0}, # 負け
            {'EntryTime': datetime(2020, 1, 5), 'ExitTime': datetime(2020, 1, 6), 'EntryPrice': 101.0, 'ExitPrice': 103.0, 'Size': 1.0, 'PnL': 2.0, 'ReturnPct': 2.0},  # 勝ち
            {'EntryTime': datetime(2020, 1, 7), 'ExitTime': datetime(2020, 1, 8), 'EntryPrice': 103.0, 'ExitPrice': 102.5, 'Size': 1.0, 'PnL': -0.5, 'ReturnPct': -0.5}, # 負け
            {'EntryTime': datetime(2020, 1, 9), 'ExitTime': datetime(2020, 1, 10), 'EntryPrice': 102.5, 'ExitPrice': 104.0, 'Size': 1.0, 'PnL': 1.5, 'ReturnPct': 1.5}, # 勝ち
        ]

        class MockTradesDF:
            def __init__(self, trades):
                self.data = trades
                self.columns = ['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'Size', 'PnL', 'ReturnPct']
                self.empty = len(trades) == 0

            def __len__(self):
                return len(self.data)

            def iterrows(self):
                for i, row in enumerate(self.data):
                    yield i, row

        mock_stats._trades = MockTradesDF(trades_data)

        # エクイティカーブ（取引による変動を反映）
        class MockEquityDF:
            def __init__(self):
                self.data = [
                    (datetime(2020, 1, 1), {'Equity': 100000.0}),
                    (datetime(2020, 1, 2), {'Equity': 100200.0}),   # +2.0
                    (datetime(2020, 1, 4), {'Equity': 100100.0}),   # -1.0
                    (datetime(2020, 1, 6), {'Equity': 100300.0}),   # +2.0
                    (datetime(2020, 1, 8), {'Equity': 100250.0}),   # -0.5
                    (datetime(2020, 1, 10), {'Equity': 100400.0}),  # +1.5
                ]
                self.columns = ['Equity']
                self.empty = False

            def __len__(self):
                return len(self.data)

            def iterrows(self):
                for timestamp, row in self.data:
                    yield timestamp, row

            @property
            def iloc(self):
                class ILocIndexer:
                    def __getitem__(self, idx):
                        if idx == 0:
                            return 100000.0
                        elif idx == -1:
                            return 100400.0  # 最終エクイティ
                        return 100000.0
                return ILocIndexer()

        mock_stats._equity_curve = MockEquityDF()

        result = converter.convert_backtest_results(
            stats=mock_stats,
            strategy_name="realistic_strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=100000.0,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 10),
            config_json=config_json
        )

        performance_metrics = result["performance_metrics"]

        # 期待される計算結果
        # 勝ちトレード: 2.0, 2.0, 1.5 = 5.5 (3勝)
        # 負けトレード: -1.0, -0.5 = -1.5 (2敗)
        # 総取引数: 5
        # 勝率: 3/5 = 60.0%
        # プロフィットファクター: 5.5 / 1.5 = 3.67
        # 平均利益: 5.5 / 3 = 1.83
        # 平均損失: 1.5 / 2 = 0.75
        # 総リターン: (100400 - 100000) / 100000 * 100 = 0.4%

        assert performance_metrics["total_trades"] == 5
        assert performance_metrics["win_rate"] == 60.0
        assert abs(performance_metrics["profit_factor"] - 3.6667) < 0.1  # 約3.67
        assert abs(performance_metrics["avg_win"] - 1.8333) < 0.1     # 約1.83
        assert performance_metrics["avg_loss"] == 0.75
        assert performance_metrics["total_return"] == 0.4
        assert performance_metrics["final_equity"] == 100400.0

        print("=== 現実的な取引シナリオテスト結果 ===")
        print(f"総取引数: {performance_metrics['total_trades']}")
        print(f"勝率: {performance_metrics['win_rate']}%")
        print(f"プロフィットファクター: {performance_metrics['profit_factor']:.2f}")
        print(f"平均利益: {performance_metrics['avg_win']:.2f}")
        print(f"平均損失: {performance_metrics['avg_loss']:.2f}")
        print(f"総リターン: {performance_metrics['total_return']:.2f}%")
        print(f"最終エクイティ: {performance_metrics['final_equity']}")