"""
設定管理パッケージ

統一設定システムを提供し、アプリケーション全体の設定を階層的に管理します。
"""

from .unified_config import (
    unified_config,
    UnifiedConfig,
    AppConfig,
    DatabaseConfig,
    LoggingConfig,
    SecurityConfig,
    MarketConfig,
    GAConfig,
    MLConfig,
)
from .validators import (
    MarketDataValidator,
    MLConfigValidator,
    DatabaseValidator,
    AppValidator,
)

# 後方互換性のため
from .settings import settings
from .market_config import MarketDataConfig

__all__ = [
    # 統一設定システム
    "unified_config",
    "UnifiedConfig",
    "AppConfig",
    "DatabaseConfig",
    "LoggingConfig",
    "SecurityConfig",
    "MarketConfig",
    "GAConfig",
    "MLConfig",
    # バリデーター
    "MarketDataValidator",
    "MLConfigValidator",
    "DatabaseValidator",
    "AppValidator",
    # 後方互換性
    "settings",
    "MarketDataConfig",
]
