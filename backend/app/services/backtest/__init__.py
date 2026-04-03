"""
バックテストサービスパッケージ

リファクタリング後のバックテスト関連サービスを提供します。
旧モジュールパスの patch/import が壊れないように互換エイリアスも公開します。
"""

import importlib as _importlib
import sys as _sys

from .config import (
    SUPPORTED_STRATEGIES,
    BacktestConfig,
    BacktestRunConfig,
    BacktestRunConfigValidationError,
    StrategyConfig,
)
from .services import BacktestDataService, BacktestService

_module_aliases = {
    "backtest_data_service": ".services.backtest_data_service",
    "backtest_service": ".services.backtest_service",
    "backtest_executor": ".execution.backtest_executor",
    "backtest_orchestrator": ".execution.backtest_orchestrator",
}

for _alias, _target in _module_aliases.items():
    _module = _importlib.import_module(_target, __name__)
    _sys.modules[f"{__name__}.{_alias}"] = _module
    setattr(_sys.modules[__name__], _alias, _module)

__all__ = [
    "BacktestConfig",
    "BacktestRunConfig",
    "BacktestRunConfigValidationError",
    "StrategyConfig",
    "SUPPORTED_STRATEGIES",
    "BacktestDataService",
    "BacktestService",
]
