"""
BacktestExecutorのテスト
"""

import warnings
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import pandas as pd

from backend.app.services.backtest.execution.backtest_executor import BacktestExecutor
from backend.app.services.backtest.execution.backtest_executor import BacktestExecutionError


class TestBacktestExecutor:
    """BacktestExecutorクラスのテスト"""

    @pytest.fixture
    def executor(self):
        """BacktestExecutorインスタンスのフィクスチャ"""
        data_service = Mock()
        return BacktestExecutor(data_service)

    def test_execute_backtest_successful(self, executor):
        """ successfulなバックテスト実行テスト"""
        # Mockデータ
        data = Mock()
        data.empty = False
        executor.data_service.get_data_for_backtest.return_value = data

        # Mock戦略クラス
        strategy_class = Mock()
        strategy_parameters = {}

        symbol = "BTCUSDT"
        timeframe = "1h"
        start_date = datetime.now()
        end_date = datetime.now()
        initial_capital = 1000000.0  # 高額でビットコイン取引パターン
        commission_rate = 0.001

        # Mockバックテスト統計結果
        stats = Mock()

        # FractionalBacktestをMock
        with patch('backend.app.services.backtest.execution.backtest_executor.FractionalBacktest') as mock_fractional_bt:
            mock_bt_instance = Mock()
            mock_bt_instance.run.return_value = stats
            mock_fractional_bt.return_value = mock_bt_instance

            result = executor.execute_backtest(
                strategy_class, strategy_parameters, symbol, timeframe,
                start_date, end_date, initial_capital, commission_rate
            )

            assert result == stats
            mock_fractional_bt.assert_called_once()
            mock_bt_instance.run.assert_called_once_with(**strategy_parameters)

    def test_execute_backtest_no_warnings_for_large_prices(self, executor, caplog):
        """高額価格のビットコインデータで警告が出ないことをテスト"""
        # Mockデータ - ビットコインのような高額価格データ
        data = pd.DataFrame({
            'Open': [95000.0, 96000.0],
            'High': [97000.0, 98000.0],
            'Low': [94000.0, 95000.0],
            'Close': [96000.0, 97000.0],
            'Volume': [100.0, 110.0]
        })
        executor.data_service.get_data_for_backtest.return_value = data

        strategy_class = Mock()
        strategy_parameters = {}

        symbol = "BTCUSDT"
        timeframe = "1h"
        start_date = datetime.now()
        end_date = datetime.now()
        initial_capital = 10000.0  # 初期資金10万ドル
        commission_rate = 0.001

        stats = Mock()

        # 警告を監視
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch('backend.app.services.backtest.execution.backtest_executor.FractionalBacktest') as mock_fractional_bt:
                mock_bt_instance = Mock()
                mock_bt_instance.run.return_value = stats
                mock_fractional_bt.return_value = mock_bt_instance

                executor.execute_backtest(
                    strategy_class, strategy_parameters, symbol, timeframe,
                    start_date, end_date, initial_capital, commission_rate
                )

                # UserWarningがキャプチャされているかチェック
                price_warnings = [warning for warning in w if issubclass(warning.category, UserWarning) and "initial cash value" in str(warning.message)]
                assert len(price_warnings) == 0, f"警告が発生しました: {[str(w.message) for w in price_warnings]}"

    def test_execute_backtest_empty_data_error(self, executor):
        """空データでのエラーテスト"""
        data = Mock()
        data.empty = True
        executor.data_service.get_data_for_backtest.return_value = data

        with pytest.raises(BacktestExecutionError, match="データが見つかりませんでした"):
            executor.execute_backtest(Mock(), {}, "BTCUSDT", "1h", datetime.now(), datetime.now(), 10000.0, 0.001)

    def test_execute_backtest_get_data_failure(self, executor):
        """データ取得失敗テスト"""
        executor.data_service.get_data_for_backtest.side_effect = Exception("Data service error")

        with pytest.raises(BacktestExecutionError, match="データ取得に失敗しました"):
            executor.execute_backtest(Mock(), {}, "BTCUSDT", "1h", datetime.now(), datetime.now(), 10000.0, 0.001)

    def test_no_crypto_detection_log_for_fractional_backtest(self, executor, caplog):
        """FractionalBacktest使用時のログ出力なしを確認"""
        import logging

        # ログレベルを設定
        caplog.set_level(logging.INFO)

        data = pd.DataFrame({
            'Open': [95000.0, 96000.0],
            'High': [97000.0, 98000.0],
            'Low': [94000.0, 95000.0],
            'Close': [96000.0, 97000.0],
            'Volume': [100.0, 110.0]
        })
        executor.data_service.get_data_for_backtest.return_value = data

        strategy_class = Mock()
        strategy_parameters = {}

        symbol = "BTC/USDT:USDT"  # 暗号通貨シンボル
        timeframe = "1h"
        start_date = datetime.now()
        end_date = datetime.now()
        initial_capital = 10000.0
        commission_rate = 0.001

        stats = Mock()

        with patch('backend.app.services.backtest.execution.backtest_executor.FractionalBacktest') as mock_fractional_bt:
            mock_bt_instance = Mock()
            mock_bt_instance.run.return_value = stats
            mock_fractional_bt.return_value = mock_bt_instance

            executor.execute_backtest(
                strategy_class, strategy_parameters, symbol, timeframe,
                start_date, end_date, initial_capital, commission_rate
            )

            # "Crypto symbol"を含むログメッセージが記録されていないことを確認
            crypto_logs = [record for record in caplog.records if "Crypto symbol" in record.message]
            assert len(crypto_logs) == 0, f"予想外のログが出力されました: {[str(record.message) for record in crypto_logs]}"

class TestFractionalBacktestIntegration:
    """FractionalBacktest統合テスト"""

    def test_fractional_backtest_used_for_crypto(self):
        """ビットコインシンボルでFractionalBacktestが使用されるテスト"""
        from backend.app.services.backtest.execution.backtest_executor import BacktestExecutor
        from backtesting.lib import FractionalBacktest
        import pandas as pd

        # 実際のビットコインデータを想定したDataFrame
        data = pd.DataFrame({
            'Open': [95000.0, 96000.0, 97000.0],
            'High': [97000.0, 98000.0, 99000.0],
            'Low': [94000.0, 95000.0, 96000.0],
            'Close': [96000.0, 97000.0, 98000.0],
            'Volume': [100.0, 110.0, 120.0]
        })

        # ダミーのStrategyクラス
        from backtesting import Strategy

        class DummyStrategy(Strategy):
            def init(self):
                pass
            def next(self):
                pass

        # FractionalBacktestを初期化
        bt = FractionalBacktest(data, DummyStrategy, cash=10000.0)

        # 警告がないことを確認
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            stats = bt.run()  # 空のrunで統計を生成

            price_warnings = [warning for warning in w if issubclass(warning.category, UserWarning) and "initial cash value" in str(warning.message)]
            assert len(price_warnings) == 0, f"警告が発生しました: {[str(w.message) for w in price_warnings]}"