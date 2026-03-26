"""
設定管理パッケージ

統一設定システムを提供し、アプリケーション全体の設定を階層的に管理します。
各設定クラスは所属サービスのモジュールに定義され、ここで一括再エクスポートされます。
"""

# 共通定数
from .constants import (
    DEFAULT_DATA_LIMIT,
    DEFAULT_ENSEMBLE_ALGORITHMS,
    DEFAULT_MARKET_EXCHANGE,
    DEFAULT_MARKET_SYMBOL,
    DEFAULT_MARKET_TIMEFRAME,
    MAX_DATA_LIMIT,
    MIN_DATA_LIMIT,
    SUPPORTED_TIMEFRAMES,
)

# コア設定
from .unified_config import (
    AppConfig,
    DatabaseConfig,
    LoggingConfig,
    MarketConfig,
    DataCollectionConfig,
    UnifiedConfig,
    unified_config,
)

# 各サービスの設定を再エクスポート
from .unified_config import (
    BacktestConfig,
    AutoStrategyConfig,
    MLConfig,
    MLDataProcessingConfig,
    MLModelConfig,
    MLPredictionConfig,
    MLTrainingConfig,
    LabelGenerationConfig,
    EnsembleConfig,
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
    # バックテスト・戦略設定
    "BacktestConfig",
    "AutoStrategyConfig",
    # ML設定
    "MLConfig",
    "MLDataProcessingConfig",
    "MLModelConfig",
    "MLPredictionConfig",
    "MLTrainingConfig",
    "LabelGenerationConfig",
    "EnsembleConfig",
]
