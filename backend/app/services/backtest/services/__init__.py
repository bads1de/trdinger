# cspell:ignore Backtest
"""
バックテストサービスパッケージ

バックテストのデータ供給と実行管理を提供するファサードサービス群です。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backtest_data_service import BacktestDataService
    from .backtest_service import BacktestService

_LAZY_EXPORTS = {
    "BacktestDataService": ".backtest_data_service",
    "BacktestService": ".backtest_service",
}

__all__ = [
    "BacktestDataService",
    "BacktestService",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _LAZY_EXPORTS)
