import warnings
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.strategies.universal_strategy import (
    StrategyEarlyTermination,
)
from app.services.backtest.execution.backtest_executor import (
    BacktestEarlyTerminationError,
    BacktestExecutionError,
    BacktestExecutor,
)


class TestBacktestExecutor:
    @pytest.fixture
    def data_service(self):
        return MagicMock()

    @pytest.fixture
    def executor(self, data_service):
        return BacktestExecutor(data_service)

    @pytest.fixture
    def sample_data(self):
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [10, 20],
            },
            index=pd.to_datetime(["2023-01-01 10:00", "2023-01-01 11:00"]),
        )
        return df

    def test_get_backtest_data_success(self, executor, data_service, sample_data):
        data_service.get_data_for_backtest.return_value = sample_data

        result = executor._get_backtest_data(
            "BTC/USDT", "1h", datetime(2023, 1, 1), datetime(2023, 1, 2)
        )

        assert not result.empty
        assert "open" in result.columns
        data_service.get_data_for_backtest.assert_called_once()

    def test_get_backtest_data_empty(self, executor, data_service):
        data_service.get_data_for_backtest.return_value = pd.DataFrame()

        with pytest.raises(BacktestExecutionError) as excinfo:
            executor._get_backtest_data(
                "BTC/USDT", "1h", datetime(2023, 1, 1), datetime(2023, 1, 2)
            )
        assert "データが見つかりませんでした" in str(excinfo.value)

    def test_create_backtest_instance(self, executor, sample_data):
        mock_strategy = MagicMock()

        # FractionalBacktestの初期化をパッチ
        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as _:
            bt = executor._create_backtest_instance(
                sample_data, mock_strategy, 10000, 0.001, 0.0005, 2.0, "BTC/USDT"
            )

            assert bt is not None

    def test_create_backtest_instance_optimization(self, executor, sample_data):
        """データコピー最適化のテスト"""
        mock_strategy = MagicMock()

        # 1. 未正規化データ（小文字）の場合
        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as MockBT:
            executor._create_backtest_instance(
                sample_data, mock_strategy, 10000, 0.0, 0.0, 1.0, "BTC"
            )
            # 渡されたデータが変換されていることを確認（モックの呼び出し引数）
            args, _ = MockBT.call_args
            passed_data = args[0]
            assert "Open" in passed_data.columns
            assert passed_data is not sample_data  # コピーされているはず

        # 2. 正規化済みデータ（大文字）の場合
        normalized_data = sample_data.copy()
        normalized_data.columns = normalized_data.columns.str.capitalize()

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as MockBT:
            executor._create_backtest_instance(
                normalized_data, mock_strategy, 10000, 0.0, 0.0, 1.0, "BTC"
            )

            # コピーされずにそのまま渡されていることを確認
            args, _ = MockBT.call_args
            passed_data = args[0]
            assert passed_data is normalized_data  # IDが一致（コピーなし）
            # カラム名が大文字になっているか確認
            args, kwargs = MockBT.call_args
            df_passed = args[0]
            assert "Open" in df_passed.columns
            assert "Close" in df_passed.columns
            assert kwargs["cash"] == 10000
            assert kwargs["commission"] == 0.0
            assert kwargs["spread"] == 0.0
            assert kwargs["margin"] == 1.0

    def test_create_backtest_instance_preserves_non_ohlcv_columns(self, executor):
        """OHLCV以外の列は正規化で壊さないことを確認する。"""
        data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [10.0, 20.0],
                "open_interest": [500.0, 510.0],
                "funding_rate": [0.01, 0.02],
                "StrategyName": ["s1", "s2"],
            }
        )
        mock_strategy = MagicMock()

        with patch(
            "app.services.backtest.execution.backtest_executor.FractionalBacktest"
        ) as MockBT:
            executor._create_backtest_instance(
                data,
                mock_strategy,
                10000,
                0.0,
                0.0,
                1.0,
                "BTC",
            )

        passed_data = MockBT.call_args.args[0]
        assert "Open" in passed_data.columns
        assert "High" in passed_data.columns
        assert "Low" in passed_data.columns
        assert "Close" in passed_data.columns
        assert "Volume" in passed_data.columns
        assert "open_interest" in passed_data.columns
        assert "funding_rate" in passed_data.columns
        assert "StrategyName" in passed_data.columns
        assert "Open_interest" not in passed_data.columns
        assert "Funding_rate" not in passed_data.columns
        assert "Strategy_name" not in passed_data.columns

    def test_run_backtest(self, executor):
        mock_bt = MagicMock()
        mock_stats = MagicMock()
        mock_bt.run.return_value = mock_stats

        params = {"p1": 10}
        result = executor._run_backtest(mock_bt, params)

        assert result == mock_stats
        mock_bt.run.assert_called_once_with(p1=10)

    def test_run_backtest_suppresses_numpy_runtimewarnings(self, executor):
        """numpy 由来の RuntimeWarning が外に漏れないこと"""

        class WarningBacktest:
            def run(self, **kwargs):
                return np.sqrt(np.array([-1.0]))

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = executor._run_backtest(WarningBacktest(), {})

        assert result is not None
        assert captured == []

    def test_run_backtest_raises_early_termination_error(self, executor):
        mock_bt = MagicMock()
        mock_bt.run.side_effect = StrategyEarlyTermination("drawdown_limit")

        with pytest.raises(BacktestEarlyTerminationError) as excinfo:
            executor._run_backtest(mock_bt, {})

        assert "drawdown_limit" in str(excinfo.value)

    def test_execute_backtest_full_flow(self, executor, data_service, sample_data):
        mock_strategy = MagicMock()
        mock_stats = MagicMock()
        data_service.get_data_for_backtest.return_value = sample_data

        with (
            patch.object(executor, "_create_backtest_instance") as mock_create,
            patch.object(
                executor, "_run_backtest", return_value=mock_stats
            ) as mock_run,
        ):

            result = executor.execute_backtest(
                mock_strategy,
                {"p1": 1},
                "BTC/USDT",
                "1h",
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                10000,
                0.001,
                0.0,
                1.0,
            )

            assert result == mock_stats
            mock_create.assert_called_once()
            mock_run.assert_called_once()

    def test_execute_backtest_preloaded(self, executor, sample_data):
        mock_strategy = MagicMock()

        with (
            patch.object(executor, "_create_backtest_instance") as mock_create,
            patch.object(executor, "_run_backtest") as mock_run,
        ):

            executor.execute_backtest(
                mock_strategy,
                {},
                "BTC/USDT",
                "1h",
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                10000,
                0.001,
                0.0,
                1.0,
                preloaded_data=sample_data,
            )

            # 内部のデータ取得が呼ばれないことを確認（data_serviceが呼ばれない）
            executor.data_service.get_data_for_backtest.assert_not_called()
            mock_create.assert_called_once()
            mock_run.assert_called_once()

    def test_get_supported_strategies(self, executor):
        strategies = executor.get_supported_strategies()
        assert "auto_strategy" in strategies
        assert strategies["auto_strategy"]["name"] == "オートストラテジー"

    def test_get_supported_strategies_uses_shared_definition(self, executor):
        sentinel = {"shared_strategy": {"name": "shared"}}

        with patch(
            "app.services.backtest.execution.backtest_executor.SUPPORTED_STRATEGIES",
            sentinel,
        ):
            strategies = executor.get_supported_strategies()

        assert strategies == sentinel
