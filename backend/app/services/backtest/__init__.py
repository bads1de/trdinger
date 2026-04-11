"""
バックテストサービスパッケージ

リファクタリング後のバックテスト関連サービスを提供します。
"""

from importlib import import_module

from .config import (
    SUPPORTED_STRATEGIES,
    BacktestConfig,
    BacktestRunConfig,
    BacktestRunConfigValidationError,
    StrategyConfig,
)
from .services import BacktestDataService, BacktestService

# 旧 import / monkeypatch パスの互換性を維持するため、モジュールも公開する。
backtest_data_service = import_module(".services.backtest_data_service", __name__)
backtest_service = import_module(".services.backtest_service", __name__)
backtest_executor = import_module(".execution.backtest_executor", __name__)
backtest_orchestrator = import_module(".execution.backtest_orchestrator", __name__)

__all__ = [
    "BacktestConfig",
    "BacktestRunConfig",
    "BacktestRunConfigValidationError",
    "StrategyConfig",
    "SUPPORTED_STRATEGIES",
    "BacktestDataService",
    "BacktestService",
    "backtest_data_service",
    "backtest_service",
    "backtest_executor",
    "backtest_orchestrator",
]
