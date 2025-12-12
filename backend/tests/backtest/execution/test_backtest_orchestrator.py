"""
BacktestOrchestratorの単体テスト
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.backtest.conversion.backtest_result_converter import (
    BacktestResultConverter,
)
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.backtest.execution.backtest_orchestrator import BacktestOrchestrator
from app.services.backtest.factories.strategy_class_factory import (
    StrategyClassFactory,
)


@pytest.fixture
def mock_data_service():
    return MagicMock(spec=BacktestDataService)


@pytest.fixture
def orchestrator(mock_data_service):
    # コンストラクタ内でのインスタンス化をモック
    with patch(
        "app.services.backtest.execution.backtest_orchestrator.StrategyClassFactory"
    ), patch(
        "app.services.backtest.execution.backtest_orchestrator.BacktestResultConverter"
    ), patch(
        "app.services.backtest.execution.backtest_orchestrator.BacktestExecutor"
    ):
        return BacktestOrchestrator(mock_data_service)


@pytest.fixture
def sample_config():
    return {
        "strategy_name": "test_strategy",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": "2023-01-01",
        "end_date": "2023-01-31",
        "initial_capital": 10000,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "GENERATED_GA",
            "parameters": {"strategy_gene": {}},
        },
    }


def test_initialization_requires_data_service():
    """初期化時にDataServiceが必須であることのテスト"""
    with pytest.raises(ValueError, match="BacktestDataService is required"):
        BacktestOrchestrator(None)


def test_run_orchestration_flow(orchestrator, sample_config):
    """
    runメソッドの実行フローテスト
    
    各コンポーネント（Validator, Factory, Executor, Converter）が
    正しい順序と引数で呼び出されることを確認
    """
    # 依存コンポーネントのモック設定
    # _validatorはPydanticに置き換わったため削除
    orchestrator._strategy_factory.create_strategy_class = MagicMock(
        return_value="StrategyClass"
    )
    orchestrator._strategy_factory.get_strategy_parameters = MagicMock(
        return_value={"param": 1}
    )
    orchestrator._executor.execute_backtest = MagicMock(return_value="Stats")
    orchestrator._result_converter.convert_backtest_results = MagicMock(
        return_value={"result": "ok"}
    )

    # 実行
    result = orchestrator.run(sample_config)

    # 検証
    # 1. バリデーション (Pydantic内で暗黙的に実行されるため、明示的な呼び出し確認は不要)

    # 2. 戦略クラス生成
    orchestrator._strategy_factory.create_strategy_class.assert_called_once()
    orchestrator._strategy_factory.get_strategy_parameters.assert_called_once()

    # 3. バックテスト実行
    orchestrator._executor.execute_backtest.assert_called_once()
    call_args = orchestrator._executor.execute_backtest.call_args[1]
    assert call_args["strategy_class"] == "StrategyClass"
    assert call_args["strategy_parameters"] == {"param": 1}
    assert call_args["symbol"] == "BTC/USDT:USDT"

    # 4. 結果変換
    orchestrator._result_converter.convert_backtest_results.assert_called_once()
    assert result == {"result": "ok"}


def test_run_date_normalization(orchestrator, sample_config):
    """日付文字列がdatetimeオブジェクトに正規化されることのテスト"""
    # モック設定
    orchestrator._executor.execute_backtest = MagicMock()
    orchestrator._result_converter.convert_backtest_results = MagicMock(
        return_value={}
    )

    # 実行
    orchestrator.run(sample_config)

    # 検証
    from datetime import datetime

    call_args = orchestrator._executor.execute_backtest.call_args[1]
    
    # _normalize_dateはdatetime.fromisoformatを使用するため、標準のdatetimeオブジェクトを返す
    assert isinstance(call_args["start_date"], datetime)
    assert isinstance(call_args["end_date"], datetime)
