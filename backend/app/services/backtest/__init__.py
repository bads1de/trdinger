"""
バックテストサービスパッケージ

リファクタリング後のバックテスト関連サービスを提供します。
"""

from importlib import import_module
from typing import Any

from .config import (
    SUPPORTED_STRATEGIES,
    BacktestConfig,
    BacktestRunConfig,
    BacktestRunConfigValidationError,
    StrategyConfig,
)

__all__ = [
    "BacktestConfig",
    "BacktestRunConfig",
    "BacktestRunConfigValidationError",
    "StrategyConfig",
    "SUPPORTED_STRATEGIES",
    "BacktestDataService",
    "BacktestService",
]

_LAZY_EXPORTS = {
    "BacktestDataService": ".services.backtest_data_service",
    "BacktestService": ".services.backtest_service",
    "backtest_data_service": ".services.backtest_data_service",
    "backtest_service": ".services.backtest_service",
    "backtest_executor": ".execution.backtest_executor",
    "backtest_orchestrator": ".execution.backtest_orchestrator",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name], __name__)
    value: Any = (
        module if name.startswith("backtest_") else getattr(module, name)
    )
    globals()[name] = value
    return value
