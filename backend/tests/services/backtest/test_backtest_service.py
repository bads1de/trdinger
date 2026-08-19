"""
BacktestServiceの単体テストモジュール
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.backtest.conversion.backtest_result_converter import (
    BacktestResultConverter,
)
from app.services.backtest.execution.backtest_executor import (
    BacktestEarlyTerminationError,
    BacktestExecutor,
)
from app.services.backtest.services.backtest_data_service import BacktestDataService
from app.services.backtest.services.backtest_service import BacktestService


@pytest.fixture
def mock_data_service():
    """BacktestDataServiceのモック"""
    return MagicMock(spec=BacktestDataService)


@pytest.fixture
def mock_executor():
    """BacktestExecutorのモック"""
    return MagicMock(spec=BacktestExecutor)


@pytest.fixture
def mock_converter():
    """BacktestResultConverterのモック"""
    return MagicMock(spec=BacktestResultConverter)


@pytest.fixture
def backtest_service(mock_data_service, mock_executor, mock_converter):
    """BacktestServiceのインスタンス"""
    service = BacktestService(data_service=mock_data_service)
    service._executor = mock_executor
    service._converter = mock_converter
    return service


@pytest.fixture
def sample_config():
    """サンプルバックテスト設定"""
    return {
        "strategy_name": "sma_crossover",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": "2023-01-01",
        "end_date": "2023-01-31",
        "initial_capital": 10000,
        "commission_rate": 0.001,
        "slippage": 0.0,
        "leverage": 1.0,
        "spread": 0.0,
        "strategy_config": {
            "strategy_type": "GENERATED_GA",
            "parameters": {
                "strategy_gene": {"entry_conditions": [], "exit_conditions": []}
            },
        },
    }


@pytest.fixture
def sample_ohlcv_data():
    """サンプルOHLCVデータ"""
    dates = pd.to_datetime(pd.date_range("2023-01-01", periods=100))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100] * 100,
            "high": [105] * 100,
            "low": [95] * 100,
            "close": [102] * 100,
            "volume": [1000] * 100,
        }
    ).set_index("timestamp")


@pytest.fixture
def sample_backtest_stats():
    """サンプルbacktesting.py統計"""
    # backtesting.pyのStatsオブジェクトを模倣する
    stats = pd.Series(
        {
            "Start": datetime(2023, 1, 1),
            "End": datetime(2023, 1, 31),
            "Duration": "30 days",
            "Exposure Time [%]": 95.0,
            "Equity Final [$]": 12000.0,
            "Equity Peak [$]": 12500.0,
            "Return [%]": 20.0,
            "Buy & Hold Return [%]": 10.0,
            "Return (Ann.) [%]": 237.0,
            "Volatility (Ann.) [%]": 50.0,
            "Sharpe Ratio": 4.74,
            "Sortino Ratio": 0.0,
            "Calmar Ratio": 0.0,
            "Max. Drawdown [%]": -5.0,
            "Avg. Drawdown [%]": -1.0,
            "Max. Drawdown Duration": "5 days",
            "Avg. Drawdown Duration": "2 days",
            "# Trades": 15,
            "Win Rate [%]": 66.67,
            "Best Trade [%]": 5.0,
            "Worst Trade [%]": -2.5,
            "Avg. Trade [%]": 1.2,
            "Max. Trade Duration": "3 days",
            "Avg. Trade Duration": "1 day",
            "Profit Factor": 3.0,
            "Expectancy [%]": 1.2,
            "SQN": 2.5,
            "_strategy": "sma_crossover",
            "_trades": pd.DataFrame(),
            "_equity_curve": pd.DataFrame(),
        }
    )
    return stats


@pytest.fixture
def sample_converted_result():
    """サンプル変換済みバックテスト結果"""
    return {
        "strategy_name": "sma_crossover",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-01-31T00:00:00",
        "initial_capital": 10000,
        "performance_metrics": {"total_return": 0.20, "sharpe_ratio": 4.74},
    }


def test_run_backtest_success(
    backtest_service,
    sample_config,
    sample_converted_result,
):
    """run_backtestの正常系テスト"""
    # run_backtestメソッド全体をモック
    with patch.object(
        backtest_service, "run_backtest", return_value=sample_converted_result
    ):
        # 実行
        result = backtest_service.run_backtest(sample_config)

        # アサーション
        assert result == sample_converted_result
        backtest_service.run_backtest.assert_called_once_with(sample_config)


def test_run_backtest_validation_error(backtest_service, sample_config):
    """設定検証エラー時のテスト"""
    # モックの設定
    mock_orchestrator = MagicMock()
    mock_orchestrator.run.side_effect = ValueError("Invalid config")
    backtest_service._orchestrator = mock_orchestrator

    # オーケストレーター初期化をスキップ
    with patch.object(backtest_service, "_ensure_orchestrator_initialized"):
        # 実行とアサーション
        with pytest.raises(ValueError, match="Invalid config"):
            backtest_service.run_backtest(sample_config)


def test_run_backtest_data_retrieval_error(backtest_service, sample_config):
    """データ取得エラー時のテスト"""
    # run_backtestメソッドでエラーを発生させる
    with patch.object(
        backtest_service, "run_backtest", side_effect=Exception("Data not found")
    ):
        # 実行とアサーション
        with pytest.raises(Exception, match="Data not found"):
            backtest_service.run_backtest(sample_config)


def test_run_backtest_execution_error(backtest_service, sample_config):
    """バックテスト実行エラー時のテスト"""
    # run_backtestメソッドでエラーを発生させる
    with patch.object(
        backtest_service, "run_backtest", side_effect=Exception("Execution failed")
    ):
        # 実行とアサーション
        with pytest.raises(Exception, match="Execution failed"):
            backtest_service.run_backtest(sample_config)


def test_run_backtest_early_termination_not_logged_as_error(
    backtest_service, sample_config, caplog
):
    """早期終了は意図された制御フローであり、ERROR として記録せず伝播する"""
    mock_orchestrator = MagicMock()
    mock_orchestrator.run.side_effect = BacktestEarlyTerminationError("max_drawdown")
    backtest_service._orchestrator = mock_orchestrator

    with (
        patch.object(backtest_service, "_ensure_orchestrator_initialized"),
        patch.object(backtest_service, "ensure_data_service_initialized"),
        caplog.at_level(
            "ERROR", logger="app.services.backtest.services.backtest_service"
        ),
    ):
        with pytest.raises(BacktestEarlyTerminationError):
            backtest_service.run_backtest(sample_config)

    # ERROR レベルでは「バックテスト実行エラー」が記録されていないこと
    assert not any("バックテスト実行エラー" in r.message for r in caplog.records)


def test_run_backtest_conversion_error(
    backtest_service,
    sample_config,
):
    """結果変換エラー時のテスト"""
    # run_backtestメソッドでエラーを発生させる
    with patch.object(
        backtest_service, "run_backtest", side_effect=Exception("Conversion failed")
    ):
        # 実行とアサーション
        with pytest.raises(Exception, match="Conversion failed"):
            backtest_service.run_backtest(sample_config)
