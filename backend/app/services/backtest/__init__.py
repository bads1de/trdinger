"""
バックテストサービスパッケージ

リファクタリング後のバックテスト関連サービスを提供します。
"""

from typing import TYPE_CHECKING

from .config import (
    SUPPORTED_STRATEGIES,
    BacktestConfig,
    BacktestRunConfig,
    BacktestRunConfigValidationError,
    StrategyConfig,
)

if TYPE_CHECKING:
    from .services.backtest_data_service import BacktestDataService
    from .services.backtest_service import BacktestService

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
}

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _LAZY_EXPORTS)
