"""
設定管理パッケージ

アプリケーション共通の設定入口を提供します。
ドメイン固有の設定は各サービス配下の config パッケージを直接参照します。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# 共通定数
from .constants import (
    DEFAULT_DATA_LIMIT,
    DEFAULT_MARKET_EXCHANGE,
    DEFAULT_MARKET_SYMBOL,
    DEFAULT_MARKET_TIMEFRAME,
    MAX_DATA_LIMIT,
    MIN_DATA_LIMIT,
    SUPPORTED_TIMEFRAMES,
)

if TYPE_CHECKING:
    # 型チェッカー/IDE には公開名を明示しつつ、
    # 実行時は __getattr__ による遅延 import を維持する。
    from .unified_config import (
        AppConfig,
        DatabaseConfig,
        DataCollectionConfig,
        LoggingConfig,
        MarketConfig,
        UnifiedConfig,
    )

_UNIFIED_CONFIG_EXPORTS = {
    "AppConfig": ".unified_config",
    "DatabaseConfig": ".unified_config",
    "LoggingConfig": ".unified_config",
    "MarketConfig": ".unified_config",
    "DataCollectionConfig": ".unified_config",
    "UnifiedConfig": ".unified_config",
}

__all__ = [
    # 共通定数
    "SUPPORTED_TIMEFRAMES",
    "DEFAULT_MARKET_EXCHANGE",
    "DEFAULT_MARKET_SYMBOL",
    "DEFAULT_MARKET_TIMEFRAME",
    "DEFAULT_DATA_LIMIT",
    "MAX_DATA_LIMIT",
    "MIN_DATA_LIMIT",
    # 統一設定システム
    "UnifiedConfig",
    # アプリケーション基本設定
    "AppConfig",
    "DatabaseConfig",
    "LoggingConfig",
    # 市場・データ収集設定
    "MarketConfig",
    "DataCollectionConfig",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _UNIFIED_CONFIG_EXPORTS)
