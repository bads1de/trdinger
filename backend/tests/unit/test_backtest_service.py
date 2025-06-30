"""
バックテスト実行サービスのテスト
"""

import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from app.core.strategies.macd_strategy import MACDStrategy


class TestBacktestService:
    """BacktestServiceのテスト"""

    @pytest.fixture
    def mock_data_service(self):
        """モックデータサービス"""
        mock_service = Mock(spec=BacktestDataService)
        return mock_service

    @pytest.fixture
    def backtest_service(self, mock_data_service):
        """BacktestServiceインスタンス"""
        service = BacktestService()
        service.data_service = mock_data_service
        return service

    @pytest.fixture
    def sample_config(self):
        """サンプルバックテスト設定"""
        return {
            "strategy_name": "MACD",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_date": datetime(2024, 12, 31, tzinfo=timezone.utc),
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "MACD",
                "parameters": {"n1": 20, "n2": 50}
            }
        }

    @pytest.fixture
    def sample_ohlcv_dataframe(self):
        """サンプルOHLCVデータフレーム"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = {
            'Open': [float(50000 + i * 10) for i in range(100)],
            'High': [float(50100 + i * 10) for i in range(100)],
            'Low': [float(49900 + i * 10) for i in range(100)],
            'Close': [float(50000 + i * 10) for i in range(100)],
            'Volume': [float(1000) for _ in range(100)]
        }
        return pd.DataFrame(data, index=dates)

    def test_run_backtest_success(self, backtest_service, mock_data_service, sample_config, sample_ohlcv_dataframe):
        """正常なバックテスト実行テスト"""
        # モックの設定
        mock_data_service.get_data_for_backtest.return_value = sample_ohlcv_dataframe

        # テスト実行
        result = backtest_service.run_backtest(sample_config)

        # 検証
        assert result is not None
        assert "strategy_name" in result
        assert "symbol" in result
        assert "timeframe" in result
        assert "performance_metrics" in result
        assert "equity_curve" in result
        assert "trade_history" in result
        assert "created_at" in result

        # パフォーマンス指標の確認
        metrics = result["performance_metrics"]
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "total_trades" in metrics

    def test_run_backtest_with_custom_strategy_parameters(self, backtest_service, mock_data_service, sample_ohlcv_dataframe):
        """カスタム戦略パラメータでのバックテストテスト"""
        config = {
            "strategy_name": "MACD",
            "symbol": "BTC/USDT",
            "timeframe": "4h",
            "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_date": datetime(2024, 6, 30, tzinfo=timezone.utc),
            "initial_capital": 50000.0,
            "commission_rate": 0.002,
            "strategy_config": {
                "strategy_type": "MACD",
                "parameters": {"n1": 10, "n2": 30}
            }
        }

        # モックの設定
        mock_data_service.get_data_for_backtest.return_value = sample_ohlcv_dataframe

        # テスト実行
        result = backtest_service.run_backtest(config)

        # カスタムパラメータが適用されていることを確認
        assert result["strategy_name"] == "MACD"
        assert result["symbol"] == "BTC/USDT"
        assert result["timeframe"] == "4h"
        assert result["initial_capital"] == 50000.0

    def test_run_backtest_no_data_error(self, backtest_service, mock_data_service, sample_config):
        """データが見つからない場合のエラーテスト"""
        # モックの設定（例外を発生させる）
        mock_data_service.get_ohlcv_for_backtest.side_effect = ValueError("No data found")

        # テスト実行とエラー確認
        with pytest.raises(ValueError, match="OHLCVデータが見つかりませんでした"):
            backtest_service.run_backtest(sample_config)

    def test_create_strategy_class_macd(self, backtest_service):
        """MACD戦略クラス作成テスト"""
        strategy_config = {
            "strategy_type": "MACD",
            "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        }

        # 戦略クラスを作成
        strategy_class = backtest_service._create_strategy_class(strategy_config)

        # 検証
        assert strategy_class is not None
        assert issubclass(strategy_class, MACDStrategy)

    def test_create_strategy_class_invalid_type(self, backtest_service):
        """無効な戦略タイプのテスト"""
        strategy_config = {
            "strategy_type": "INVALID_STRATEGY",
            "parameters": {}
        }

        # エラーが発生することを確認
        with pytest.raises(ValueError, match="サポートされていない戦略タイプ"):
            backtest_service._create_strategy_class(strategy_config)

    def test_convert_backtest_results(self, backtest_service):
        """バックテスト結果変換テスト"""
        # モックのbacktesting.py結果
        mock_stats = pd.Series({
            'Return [%]': 25.5,
            'Sharpe Ratio': 1.2,
            'Max. Drawdown [%]': -15.3,
            'Win Rate [%]': 65.0,
            '# Trades': 45,
            'Equity Final [$]': 125500.0
        })

        # 資産曲線のモック
        equity_curve = pd.DataFrame({
            'Equity': [100000, 101000, 102000],
            'DrawdownPct': [0.0, -0.01, -0.005]
        }, index=pd.date_range('2024-01-01', periods=3))

        # 取引履歴のモック
        trades = pd.DataFrame({
            'Size': [1.0, -1.0],
            'EntryPrice': [50000, 51000],
            'ExitPrice': [51000, 50500],
            'PnL': [1000, -500],
            'ReturnPct': [0.02, -0.01]
        })

        mock_stats['_equity_curve'] = equity_curve
        mock_stats['_trades'] = trades

        # 変換実行
        result = backtest_service._convert_backtest_results(
            mock_stats, "MACD", "BTC/USDT", "1h", 100000.0
        )

        # 検証
        assert result["strategy_name"] == "MACD"
        assert result["symbol"] == "BTC/USDT"
        assert result["timeframe"] == "1h"
        assert result["initial_capital"] == 100000.0

        # パフォーマンス指標
        metrics = result["performance_metrics"]
        assert metrics["total_return"] == 25.5
        assert metrics["sharpe_ratio"] == 1.2
        assert metrics["max_drawdown"] == -15.3
        assert metrics["win_rate"] == 65.0
        assert metrics["total_trades"] == 45

        # 資産曲線
        assert len(result["equity_curve"]) == 3
        assert result["equity_curve"][0]["equity"] == 100000

        # 取引履歴
        assert len(result["trade_history"]) == 2
        assert result["trade_history"][0]["pnl"] == 1000

    def test_validate_backtest_config(self, backtest_service):
        """バックテスト設定の検証テスト"""
        # 有効な設定
        valid_config = {
            "strategy_name": "MACD",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_date": datetime(2024, 12, 31, tzinfo=timezone.utc),
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "MACD",
                "parameters": {"n1": 20, "n2": 50}
            }
        }

        # 検証が成功することを確認
        assert backtest_service._validate_config(valid_config) is True

    def test_validate_backtest_config_invalid_dates(self, backtest_service):
        """無効な日付設定の検証テスト"""
        invalid_config = {
            "strategy_name": "MACD",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": datetime(2024, 12, 31, tzinfo=timezone.utc),  # 終了日より後
            "end_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "MACD",
                "parameters": {"n1": 20, "n2": 50}
            }
        }

        # エラーが発生することを確認
        with pytest.raises(ValueError, match="開始日は終了日よりも前である必要があります"):
            backtest_service._validate_config(invalid_config)

    def test_validate_backtest_config_invalid_capital(self, backtest_service):
        """無効な初期資金の検証テスト"""
        invalid_config = {
            "strategy_name": "MACD",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_date": datetime(2024, 12, 31, tzinfo=timezone.utc),
            "initial_capital": -1000.0,  # 負の値
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "MACD",
                "parameters": {"n1": 20, "n2": 50}
            }
        }

        # エラーが発生することを確認
        with pytest.raises(ValueError, match="初期資金は正の数である必要があります"):
            backtest_service._validate_config(invalid_config)

    @patch('app.core.services.backtest_service.Backtest')
    def test_run_backtest_with_backtesting_py(self, mock_backtest_class, backtest_service, mock_data_service, sample_config, sample_ohlcv_dataframe):
        """backtesting.pyライブラリとの統合テスト"""
        # モックの設定
        mock_data_service.get_data_for_backtest.return_value = sample_ohlcv_dataframe
        
        mock_backtest_instance = Mock()
        mock_backtest_class.return_value = mock_backtest_instance
        
        # モックの統計結果
        mock_stats = pd.Series({
            'Return [%]': 15.0,
            'Sharpe Ratio': 0.8,
            'Max. Drawdown [%]': -10.0,
            'Win Rate [%]': 55.0,
            '# Trades': 20,
            'Equity Final [$]': 115000.0,
            '_equity_curve': pd.DataFrame({'Equity': [100000, 115000]}),
            '_trades': pd.DataFrame({'PnL': [1000, -500]})
        })
        mock_backtest_instance.run.return_value = mock_stats

        # テスト実行
        result = backtest_service.run_backtest(sample_config)

        # Backtestクラスが正しいパラメータで呼び出されたことを確認
        mock_backtest_class.assert_called_once()
        call_args = mock_backtest_class.call_args
        
        # データフレームが渡されていることを確認
        assert isinstance(call_args[0][0], pd.DataFrame)
        
        # 戦略クラスが渡されていることを確認
        assert issubclass(call_args[0][1], MACDStrategy)
        
        # キーワード引数の確認
        kwargs = call_args[1]
        assert kwargs['cash'] == 100000.0
        assert kwargs['commission'] == 0.001
        assert kwargs['exclusive_orders'] is True

        # run()メソッドが正しいパラメータで呼ばれたことを確認
        expected_params = sample_config.get("strategy_config", {}).get("parameters", {})
        mock_backtest_instance.run.assert_called_once_with(**expected_params)

        # 結果の確認
        assert result["performance_metrics"]["total_return"] == 15.0
