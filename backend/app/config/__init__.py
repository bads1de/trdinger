"""
設定管理パッケージ

アプリケーション共通の設定入口を提供します。
ドメイン固有の設定は各サービス配下の config パッケージを直接参照します。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# 共通定数
from .constants import (  # noqa: F401
    DEFAULT_DATA_LIMIT,
    DEFAULT_ENSEMBLE_ALGORITHMS,
    DEFAULT_MARKET_EXCHANGE,
    DEFAULT_MARKET_SYMBOL,
    DEFAULT_MARKET_TIMEFRAME,
    MAX_DATA_LIMIT,
    MIN_DATA_LIMIT,
    SUPPORTED_TIMEFRAMES,
)

_UNIFIED_CONFIG_EXPORTS = {
    "AppConfig",
    "DatabaseConfig",
    "LoggingConfig",
    "MarketConfig",
    "DataCollectionConfig",
    "UnifiedConfig",
    "unified_config",
}


def __getattr__(name: str) -> Any:
    if name in _UNIFIED_CONFIG_EXPORTS:
        module = import_module(".unified_config", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        {
            *globals().keys(),
            *_UNIFIED_CONFIG_EXPORTS,
        }
    )


__all__ = [
    # 共通定数
    "SUPPORTED_TIMEFRAMES",
    "DEFAULT_ENSEMBLE_ALGORITHMS",
    "DEFAULT_MARKET_EXCHANGE",
    "DEFAULT_MARKET_SYMBOL",
    "DEFAULT_MARKET_TIMEFRAME",
    "DEFAULT_DATA_LIMIT",
    "MAX_DATA_LIMIT",
    "MIN_DATA_LIMIT",
    # 統一設定システム
    "unified_config",
    "UnifiedConfig",
    # アプリケーション基本設定
    "AppConfig",
    "DatabaseConfig",
    "LoggingConfig",
    # 市場・データ収集設定
    "MarketConfig",
    "DataCollectionConfig",
]
