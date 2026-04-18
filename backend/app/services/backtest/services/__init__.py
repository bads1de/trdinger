"""
バックテストサービスパッケージ

バックテストのデータ供給と実行管理を提供するファサードサービス群です。
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "BacktestDataService": ".backtest_data_service",
    "BacktestService": ".backtest_service",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name], __name__)
    value: Any = getattr(module, name)
    globals()[name] = value
    return value
