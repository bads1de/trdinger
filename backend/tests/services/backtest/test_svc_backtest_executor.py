"""
バックテスト実行エンジンテスト

BacktestExecutorの機能をテストします。
"""

import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from backtesting import Strategy

from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.backtest.execution.backtest_executor import (
    BacktestExecutionError,
    BacktestExecutor,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_data_service():
    """モックデータサービス"""
    return MagicMock(spec=BacktestDataService)


@pytest.fixture
def backtest_executor(mock_data_service):
    """BacktestExecutorインスタンス"""
    return BacktestExecutor(data_service=mock_data_service)


@pytest.fixture
def sample_backtest_data():
    """テスト用バックテストデータ（大文字カラム名）"""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "Open": [100.0 + i * 0.1 for i in range(100)],
            "High": [105.0 + i * 0.1 for i in range(100)],
            "Low": [95.0 + i * 0.1 for i in range(100)],
            "Close": [102.0 + i * 0.1 for i in range(100)],
            "Volume": [1000.0 + i * 10 for i in range(100)],
        },
        index=dates,
    )


@pytest.fixture
def sample_strategy_class():
    """テスト用戦略クラス"""

    class TestStrategy(Strategy):
        def init(self):
            pass

        def next(self):
            if len(self.data) > 0 and not self.position:
                self.buy()
            elif self.position:
                self.position.close()

    return TestStrategy


@pytest.fixture
def mock_backtest_stats():
    """モックバックテスト統計"""
    stats = pd.Series(
        {
            "Return [%]": 10.5,
            "# Trades": 5,
            "Win Rate [%]": 60.0,
            "Profit Factor": 1.8,
            "Best Trade [%]": 5.2,
            "Worst Trade [%]": -2.1,
            "Max. Drawdown [%]": -8.5,
            "Sharpe Ratio": 1.2,
            "Equity Final [$]": 11050.0,
        }
    )
    stats._trades = pd.DataFrame()
    stats._equity_curve = pd.DataFrame()
    return stats


class TestExecutorInitialization:
    """実行エンジン初期化テスト"""

    def test_initialize_with_data_service(self, mock_data_service):
        """データサービス付きで初期化できること"""
        executor = BacktestExecutor(data_service=mock_data_service)

        assert executor.data_service == mock_data_service

    def test_initialize_requires_data_service(self):
        """データサービスなしでは初期化できないこと"""
        # データサービスは必須パラメータ
        with pytest.raises(TypeError):
            BacktestExecutor()


class TestBacktestExecution:
    """バックテスト実行テスト"""

    def test_execute_backtest_success(
        self,
        backtest_executor,
        mock_data_service,
        sample_backtest_data,
        sample_strategy_class,
        mock_backtest_stats,
    ):
        """バックテストを正常に実行できること"""
        mock_data_service.get_data_for_backtest.return_value = sample_backtest_data

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt = MagicMock()
            mock_bt.run.return_value = mock_backtest_stats
            mock_bt_class.return_value = mock_bt

            result = backtest_executor.execute_backtest(
                strategy_class=sample_strategy_class,
                strategy_parameters={"param1": "value1"},
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
                initial_capital=10000.0,
                commission_rate=0.001,
            )

            assert result is not None
            mock_data_service.get_data_for_backtest.assert_called_once()
            mock_bt.run.assert_called_once_with(param1="value1")

    def test_execute_backtest_with_parameters(
        self,
        backtest_executor,
        mock_data_service,
        sample_backtest_data,
        sample_strategy_class,
        mock_backtest_stats,
    ):
        """戦略パラメータ付きでバックテストを実行できること"""
        mock_data_service.get_data_for_backtest.return_value = sample_backtest_data

        strategy_params = {
            "fast_period": 10,
            "slow_period": 30,
            "threshold": 0.02,
        }

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt = MagicMock()
            mock_bt.run.return_value = mock_backtest_stats
            mock_bt_class.return_value = mock_bt

            result = backtest_executor.execute_backtest(
                strategy_class=sample_strategy_class,
                strategy_parameters=strategy_params,
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
                initial_capital=10000.0,
                commission_rate=0.001,
            )

            assert result is not None
            mock_bt.run.assert_called_once_with(**strategy_params)

    def test_execute_backtest_data_retrieval_error(
        self,
        backtest_executor,
        mock_data_service,
        sample_strategy_class,
    ):
        """データ取得エラー時の処理"""
        mock_data_service.get_data_for_backtest.side_effect = Exception(
            "データベースエラー"
        )

        with pytest.raises(BacktestExecutionError, match="データ取得に失敗しました"):
            backtest_executor.execute_backtest(
                strategy_class=sample_strategy_class,
                strategy_parameters={},
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
                initial_capital=10000.0,
                commission_rate=0.001,
            )

    def test_execute_backtest_empty_data(
        self,
        backtest_executor,
        mock_data_service,
        sample_strategy_class,
    ):
        """空データでのバックテスト実行エラー"""
        mock_data_service.get_data_for_backtest.return_value = pd.DataFrame()

        with pytest.raises(
            BacktestExecutionError, match="のデータが見つかりませんでした"
        ):
            backtest_executor.execute_backtest(
                strategy_class=sample_strategy_class,
                strategy_parameters={},
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
                initial_capital=10000.0,
                commission_rate=0.001,
            )


class TestDataPreparation:
    """データ準備テスト"""

    def test_get_backtest_data_success(
        self, backtest_executor, mock_data_service, sample_backtest_data
    ):
        """バックテスト用データを正常に取得できること"""
        mock_data_service.get_data_for_backtest.return_value = sample_backtest_data

        result = backtest_executor._get_backtest_data(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
        )

        assert not result.empty
        assert len(result) == 100
        mock_data_service.get_data_for_backtest.assert_called_once_with(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
        )

    def test_get_backtest_data_empty_result(self, backtest_executor, mock_data_service):
        """空のデータ取得時のエラー処理"""
        mock_data_service.get_data_for_backtest.return_value = pd.DataFrame()

        with pytest.raises(
            BacktestExecutionError, match="のデータが見つかりませんでした"
        ):
            backtest_executor._get_backtest_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

    def test_get_backtest_data_service_error(
        self, backtest_executor, mock_data_service
    ):
        """データサービスエラー時の処理"""
        mock_data_service.get_data_for_backtest.side_effect = ValueError(
            "データ取得失敗"
        )

        with pytest.raises(BacktestExecutionError, match="データ取得に失敗しました"):
            backtest_executor._get_backtest_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )


class TestBacktestInstanceCreation:
    """バックテストインスタンス作成テスト"""

    def test_create_backtest_instance_success(
        self, backtest_executor, sample_backtest_data, sample_strategy_class
    ):
        """バックテストインスタンスを正常に作成できること"""
        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt = MagicMock()
            mock_bt_class.return_value = mock_bt

            result = backtest_executor._create_backtest_instance(
                data=sample_backtest_data,
                strategy_class=sample_strategy_class,
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage=0.0,
                leverage=1.0,
                symbol="BTC/USDT:USDT",
            )

            assert result == mock_bt
            mock_bt_class.assert_called_once()

    def test_create_backtest_instance_column_capitalization(
        self, backtest_executor, sample_strategy_class
    ):
        """カラム名が大文字に変換されること"""
        # 小文字カラム名のデータ
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        lowercase_data = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [105.0] * 10,
                "low": [95.0] * 10,
                "close": [102.0] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt = MagicMock()
            mock_bt_class.return_value = mock_bt

            backtest_executor._create_backtest_instance(
                data=lowercase_data,
                strategy_class=sample_strategy_class,
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage=0.0,
                leverage=1.0,
                symbol="BTC/USDT:USDT",
            )

            # FractionalBacktestに渡されたデータのカラムが大文字であることを確認
            call_args = mock_bt_class.call_args
            data_arg = call_args[0][0]
            assert "Open" in data_arg.columns
            assert "High" in data_arg.columns
            assert "Low" in data_arg.columns
            assert "Close" in data_arg.columns
            assert "Volume" in data_arg.columns

    def test_create_backtest_instance_with_settings(
        self, backtest_executor, sample_backtest_data, sample_strategy_class
    ):
        """適切な設定でバックテストインスタンスが作成されること"""
        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt = MagicMock()
            mock_bt_class.return_value = mock_bt

            backtest_executor._create_backtest_instance(
                data=sample_backtest_data,
                strategy_class=sample_strategy_class,
                initial_capital=50000.0,
                commission_rate=0.002,
                slippage=0.0001,
                leverage=10.0,
                symbol="ETH/USDT",
            )

            # FractionalBacktestが適切なパラメータで呼ばれたことを確認
            call_kwargs = mock_bt_class.call_args[1]
            assert call_kwargs["cash"] == 50000.0
            assert call_kwargs["commission"] == 0.0021
            assert call_kwargs["exclusive_orders"] is False
            assert call_kwargs["trade_on_close"] is False
            assert call_kwargs["hedging"] is True
            assert call_kwargs["margin"] == 0.1

    def test_create_backtest_instance_error(
        self, backtest_executor, sample_backtest_data, sample_strategy_class
    ):
        """バックテストインスタンス作成エラー時の処理"""
        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest",
            side_effect=Exception("インスタンス作成失敗"),
        ):
            with pytest.raises(
                BacktestExecutionError,
                match="バックテストインスタンスの作成に失敗しました",
            ):
                backtest_executor._create_backtest_instance(
                    data=sample_backtest_data,
                    strategy_class=sample_strategy_class,
                    initial_capital=10000.0,
                    commission_rate=0.001,
                    slippage=0.0,
                    leverage=1.0,
                    symbol="BTC/USDT:USDT",
                )


class TestBacktestRun:
    """バックテスト実行テスト"""

    def test_run_backtest_success(self, backtest_executor, mock_backtest_stats):
        """バックテストを正常に実行できること"""
        mock_bt = MagicMock()
        mock_bt.run.return_value = mock_backtest_stats

        result = backtest_executor._run_backtest(
            bt=mock_bt, strategy_parameters={"param1": "value1"}
        )

        assert result is not None
        mock_bt.run.assert_called_once_with(param1="value1")

    def test_run_backtest_with_multiple_parameters(
        self, backtest_executor, mock_backtest_stats
    ):
        """複数パラメータでバックテストを実行できること"""
        mock_bt = MagicMock()
        mock_bt.run.return_value = mock_backtest_stats

        params = {
            "fast_period": 10,
            "slow_period": 30,
            "stop_loss": 0.02,
            "take_profit": 0.05,
        }

        result = backtest_executor._run_backtest(bt=mock_bt, strategy_parameters=params)

        assert result is not None
        mock_bt.run.assert_called_once_with(**params)

    def test_run_backtest_empty_parameters(
        self, backtest_executor, mock_backtest_stats
    ):
        """パラメータなしでバックテストを実行できること"""
        mock_bt = MagicMock()
        mock_bt.run.return_value = mock_backtest_stats

        result = backtest_executor._run_backtest(bt=mock_bt, strategy_parameters={})

        assert result is not None
        mock_bt.run.assert_called_once_with()

    def test_run_backtest_execution_error(self, backtest_executor):
        """バックテスト実行エラー時の処理"""
        mock_bt = MagicMock()
        mock_bt.run.side_effect = Exception("実行エラー")

        with pytest.raises(
            BacktestExecutionError, match="バックテスト実行中にエラーが発生しました"
        ):
            backtest_executor._run_backtest(
                bt=mock_bt, strategy_parameters={"param1": "value1"}
            )

    def test_run_backtest_warnings_suppressed(
        self, backtest_executor, mock_backtest_stats
    ):
        """警告が抑制されること"""
        mock_bt = MagicMock()
        mock_bt.run.return_value = mock_backtest_stats

        # 警告を発生させる
        import warnings

        with patch.object(warnings, "filterwarnings") as mock_filter:
            backtest_executor._run_backtest(
                bt=mock_bt, strategy_parameters={"param1": "value1"}
            )

            # 警告フィルタが呼ばれたことを確認
            mock_filter.assert_called()


class TestSupportedStrategies:
    """サポート戦略テスト"""

    def test_get_supported_strategies(self, backtest_executor):
        """サポートされている戦略一覧を取得できること"""
        strategies = backtest_executor.get_supported_strategies()

        assert "auto_strategy" in strategies
        assert "name" in strategies["auto_strategy"]
        assert "description" in strategies["auto_strategy"]
        assert "parameters" in strategies["auto_strategy"]

    def test_supported_strategies_structure(self, backtest_executor):
        """サポート戦略の構造が正しいこと"""
        strategies = backtest_executor.get_supported_strategies()
        auto_strategy = strategies["auto_strategy"]

        assert auto_strategy["name"] == "オートストラテジー"
        assert "遺伝的アルゴリズム" in auto_strategy["description"]
        assert "strategy_gene" in auto_strategy["parameters"]

        strategy_gene_param = auto_strategy["parameters"]["strategy_gene"]
        assert strategy_gene_param["type"] == "dict"
        assert strategy_gene_param["required"] is True


class TestPerformanceAndEdgeCases:
    """パフォーマンスとエッジケーステスト"""

    def test_execute_backtest_with_large_dataset(
        self,
        backtest_executor,
        mock_data_service,
        sample_strategy_class,
        mock_backtest_stats,
    ):
        """大量データでのバックテスト実行"""
        # 1年分のデータ（8760時間）
        dates = pd.date_range("2024-01-01", periods=8760, freq="h")
        large_data = pd.DataFrame(
            {
                "Open": [100.0 + i * 0.01 for i in range(8760)],
                "High": [105.0 + i * 0.01 for i in range(8760)],
                "Low": [95.0 + i * 0.01 for i in range(8760)],
                "Close": [102.0 + i * 0.01 for i in range(8760)],
                "Volume": [1000.0 + i for i in range(8760)],
            },
            index=dates,
        )

        mock_data_service.get_data_for_backtest.return_value = large_data

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt = MagicMock()
            mock_bt.run.return_value = mock_backtest_stats
            mock_bt_class.return_value = mock_bt

            result = backtest_executor.execute_backtest(
                strategy_class=sample_strategy_class,
                strategy_parameters={},
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                initial_capital=10000.0,
                commission_rate=0.001,
            )

            assert result is not None

    def test_execute_backtest_with_minimal_capital(
        self,
        backtest_executor,
        mock_data_service,
        sample_backtest_data,
        sample_strategy_class,
        mock_backtest_stats,
    ):
        """最小資金でのバックテスト実行"""
        mock_data_service.get_data_for_backtest.return_value = sample_backtest_data

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt = MagicMock()
            mock_bt.run.return_value = mock_backtest_stats
            mock_bt_class.return_value = mock_bt

            result = backtest_executor.execute_backtest(
                strategy_class=sample_strategy_class,
                strategy_parameters={},
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
                initial_capital=100.0,  # 最小資金
                commission_rate=0.001,
            )

            assert result is not None

    def test_execute_backtest_with_high_commission(
        self,
        backtest_executor,
        mock_data_service,
        sample_backtest_data,
        sample_strategy_class,
        mock_backtest_stats,
    ):
        """高い手数料率でのバックテスト実行"""
        mock_data_service.get_data_for_backtest.return_value = sample_backtest_data

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt = MagicMock()
            mock_bt.run.return_value = mock_backtest_stats
            mock_bt_class.return_value = mock_bt

            result = backtest_executor.execute_backtest(
                strategy_class=sample_strategy_class,
                strategy_parameters={},
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
                initial_capital=10000.0,
                commission_rate=0.01,  # 1%の高い手数料
            )

            assert result is not None

    def test_execute_backtest_with_short_period(
        self,
        backtest_executor,
        mock_data_service,
        sample_strategy_class,
        mock_backtest_stats,
    ):
        """短期間でのバックテスト実行"""
        dates = pd.date_range("2024-01-01", periods=24, freq="h")  # 1日分
        short_data = pd.DataFrame(
            {
                "Open": [100.0] * 24,
                "High": [105.0] * 24,
                "Low": [95.0] * 24,
                "Close": [102.0] * 24,
                "Volume": [1000.0] * 24,
            },
            index=dates,
        )

        mock_data_service.get_data_for_backtest.return_value = short_data

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt = MagicMock()
            mock_bt.run.return_value = mock_backtest_stats
            mock_bt_class.return_value = mock_bt

            result = backtest_executor.execute_backtest(
                strategy_class=sample_strategy_class,
                strategy_parameters={},
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
                initial_capital=10000.0,
                commission_rate=0.001,
            )

            assert result is not None


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_handle_invalid_strategy_class(
        self,
        backtest_executor,
        mock_data_service,
        sample_backtest_data,
    ):
        """無効な戦略クラスでのエラー処理"""
        mock_data_service.get_data_for_backtest.return_value = sample_backtest_data

        # 戦略でない無効なクラス
        class InvalidStrategy:
            pass

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt_class.side_effect = Exception("無効な戦略クラス")

            with pytest.raises(BacktestExecutionError):
                backtest_executor.execute_backtest(
                    strategy_class=InvalidStrategy,
                    strategy_parameters={},
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 5),
                    initial_capital=10000.0,
                    commission_rate=0.001,
                )

    def test_handle_data_format_error(
        self,
        backtest_executor,
        mock_data_service,
        sample_strategy_class,
    ):
        """不正なデータフォーマットでのエラー処理"""
        # カラムが不足しているデータ
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        invalid_data = pd.DataFrame(
            {
                "Open": [100.0] * 10,
                "Close": [102.0] * 10,
                # High, Low, Volumeが欠けている
            },
            index=dates,
        )

        mock_data_service.get_data_for_backtest.return_value = invalid_data

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt_class.side_effect = KeyError("必須カラムが不足しています")

            with pytest.raises(BacktestExecutionError):
                backtest_executor.execute_backtest(
                    strategy_class=sample_strategy_class,
                    strategy_parameters={},
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 5),
                    initial_capital=10000.0,
                    commission_rate=0.001,
                )

    def test_handle_negative_capital(
        self,
        backtest_executor,
        mock_data_service,
        sample_backtest_data,
        sample_strategy_class,
    ):
        """負の初期資金でのエラー処理"""
        mock_data_service.get_data_for_backtest.return_value = sample_backtest_data

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as mock_bt_class:
            mock_bt_class.side_effect = ValueError(
                "初期資金は正の数値である必要があります"
            )

            with pytest.raises(BacktestExecutionError):
                backtest_executor.execute_backtest(
                    strategy_class=sample_strategy_class,
                    strategy_parameters={},
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 5),
                    initial_capital=-1000.0,  # 負の値
                    commission_rate=0.001,
                )
