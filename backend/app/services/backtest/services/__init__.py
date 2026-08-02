"""
バックテストサービスパッケージ

バックテストのデータ供給と実行管理を提供するファサードサービス群です。
"""

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
