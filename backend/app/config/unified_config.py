"""
統一設定システム

アプリケーション全体の設定を階層的で管理しやすい構造で統一管理します。
SOLID原則に従い、各設定カテゴリを明確に分離し、責任を明確化します。
"""

from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 共通定数
from app.config.constants import (  # noqa: F401
    DEFAULT_DATA_LIMIT,
    DEFAULT_ENSEMBLE_ALGORITHMS,
    DEFAULT_MARKET_EXCHANGE,
    DEFAULT_MARKET_SYMBOL,
    DEFAULT_MARKET_TIMEFRAME,
    MAX_DATA_LIMIT,
    MIN_DATA_LIMIT,
    SUPPORTED_TIMEFRAMES,
)

# 各サービスの環境変数設定をインポート（サービス側の実体を正本として参照）
from app.services.backtest.config.backtest_settings import BacktestConfig  # noqa: F401
from app.services.auto_strategy.config.auto_strategy_settings import (  # noqa: F401
    AutoStrategyConfig,
)
from app.services.ml.common.ml_config import (  # noqa: F401
    EnsembleConfig,
    LabelGenerationConfig,
    MLConfig,
    MLDataProcessingConfig,
    MLModelConfig,
    MLPredictionConfig,
    MLTrainingConfig,
)


class AppConfig(BaseSettings):
    """アプリケーション基本設定。

    アプリケーションの基本的な設定を管理します。CORS設定やデバッグモードなどを含みます。
    """

    app_name: str = Field(default="Trdinger Trading API", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"], alias="CORS_ORIGINS"
    )

    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")


class DatabaseConfig(BaseSettings):
    """データベース設定。

    PostgreSQLデータベース接続の設定を管理します。
    """

    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    host: str = Field(default="localhost", alias="DB_HOST")
    port: int = Field(default=5432, alias="DB_PORT")
    name: str = Field(default="trdinger", alias="DB_NAME")
    user: str = Field(default="postgres", alias="DB_USER")
    password: str = Field(default="", alias="DB_PASSWORD")

    @property
    def url_complete(self) -> str:
        """完全なデータベースURLを生成します。

        DATABASE_URLが設定されている場合はそれを返し、
        そうでない場合は個別パラメータからURLを構築します。

        Returns:
            str: 完全なデータベース接続URL。
        """
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")


class LoggingConfig(BaseSettings):
    """ログ設定。

    ロギングのレベル、フォーマット、ファイル出力などの設定を管理します。
    """

    level: str = Field(default="DEBUG", alias="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="LOG_FORMAT",
    )
    file: str = Field(default="market_data.log", alias="LOG_FILE")
    max_bytes: int = Field(default=10485760, alias="LOG_MAX_BYTES")

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")


class MarketConfig(BaseSettings):
    """市場データ設定。

    取引所、シンボル、時間軸などの市場データ関連の設定を管理します。
    """

    # 基本設定
    sandbox: bool = Field(default=False, alias="MARKET_DATA_SANDBOX")
    enable_cache: bool = Field(default=True, alias="ENABLE_CACHE")
    max_cache_size: int = Field(default=MAX_DATA_LIMIT, alias="MAX_CACHE_SIZE")

    # サポートされている取引所
    supported_exchanges: List[str] = Field(
        default_factory=lambda: [DEFAULT_MARKET_EXCHANGE]
    )

    # サポートされているシンボル（Bybit形式）
    supported_symbols: List[str] = Field(
        default_factory=lambda: [DEFAULT_MARKET_SYMBOL]
    )

    # サポートされている時間軸
    supported_timeframes: List[str] = Field(
        default_factory=lambda: list(SUPPORTED_TIMEFRAMES)
    )

    # デフォルト設定
    default_exchange: str = Field(default=DEFAULT_MARKET_EXCHANGE)
    default_symbol: str = Field(default=DEFAULT_MARKET_SYMBOL)
    default_timeframe: str = Field(default=DEFAULT_MARKET_TIMEFRAME)
    default_limit: int = Field(default=DEFAULT_DATA_LIMIT, description="デフォルト取得件数")
    max_limit: int = Field(default=MAX_DATA_LIMIT, description="最大取得件数")
    min_limit: int = Field(default=MIN_DATA_LIMIT, description="最小取得件数")

    # Bybit固有の設定
    bybit_config: Dict[str, Any] = Field(
        default={
            "sandbox": False,
            "enableRateLimit": True,
            "timeout": 30000,
        }
    )

    # シンボル正規化マッピング
    symbol_mapping: Dict[str, str] = Field(
        default={
            "BTCUSDT": "BTC/USDT:USDT",
            "BTC-USDT": "BTC/USDT:USDT",
            "BTCUSDT_PERP": "BTC/USDT:USDT",
        }
    )

    model_config = SettingsConfigDict(env_prefix="MARKET_", extra="ignore")


class DataCollectionConfig(BaseSettings):
    """データ収集設定。

    市場データ収集のAPI制限、タイムアウト、メモリ管理などの設定を管理します。
    """

    # API制限設定
    default_limit: int = Field(default=DEFAULT_DATA_LIMIT, description="デフォルト取得件数")
    max_limit: int = Field(default=MAX_DATA_LIMIT, description="最大取得件数")
    min_limit: int = Field(default=MIN_DATA_LIMIT, description="最小取得件数")

    # Bybit API設定
    bybit_timeout: int = Field(default=30, description="APIタイムアウト（秒）")
    bybit_page_limit: int = Field(default=200, description="ページング時の制限")
    bybit_max_pages: int = Field(
        default=500, description="最大ページ数（2020年からのデータを取得可能に拡張）"
    )

    # メモリ管理
    memory_warning_threshold: int = Field(default=8000, description="メモリ警告閾値")
    memory_limit_threshold: int = Field(default=10000, description="メモリ制限閾値")

    model_config = SettingsConfigDict(env_prefix="DATA_COLLECTION_", extra="ignore")


class UnifiedConfig(BaseSettings):
    """アプリケーション全体の統一設定クラス。

    全ての設定を階層的に管理し、環境変数からの設定読み込みをサポートします。
    """

    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    auto_strategy: AutoStrategyConfig = Field(default_factory=AutoStrategyConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",  # ネストされた環境変数をサポート
    )


# 統一設定のシングルトンインスタンス
unified_config = UnifiedConfig()
