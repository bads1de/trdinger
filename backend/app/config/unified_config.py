"""
統一設定システム

アプリケーション全体の設定を階層的で管理しやすい構造で統一管理します。
SOLID原則に従い、各設定カテゴリを明確に分離し、責任を明確化します。
"""

from typing import Optional, List, Dict, Any
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings
import os


class AppConfig(BaseSettings):
    """アプリケーション基本設定"""

    app_name: str = Field(default="Trdinger Trading API", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"], alias="CORS_ORIGINS"
    )

    class Config:
        env_prefix = "APP_"
        extra = "ignore"


class DatabaseConfig(BaseSettings):
    """データベース設定"""

    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    host: str = Field(default="localhost", alias="DB_HOST")
    port: int = Field(default=5432, alias="DB_PORT")
    name: str = Field(default="trdinger", alias="DB_NAME")
    user: str = Field(default="postgres", alias="DB_USER")
    password: str = Field(default="", alias="DB_PASSWORD")

    @property
    def url_complete(self) -> str:
        """完全なデータベースURLを生成"""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    class Config:
        env_prefix = "DB_"
        extra = "ignore"


class LoggingConfig(BaseSettings):
    """ログ設定"""

    level: str = Field(default="DEBUG", alias="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="LOG_FORMAT",
    )
    file: str = Field(default="market_data.log", alias="LOG_FILE")
    max_bytes: int = Field(default=10485760, alias="LOG_MAX_BYTES")

    class Config:
        env_prefix = "LOG_"
        extra = "ignore"


class SecurityConfig(BaseSettings):
    """セキュリティ設定"""

    secret_key: str = Field(default="your-secret-key-here", alias="SECRET_KEY")

    class Config:
        env_prefix = "SECURITY_"
        extra = "ignore"


class MarketConfig(BaseSettings):
    """市場データ設定"""

    # 基本設定
    sandbox: bool = Field(default=False, alias="MARKET_DATA_SANDBOX")
    enable_cache: bool = Field(default=True, alias="ENABLE_CACHE")
    max_cache_size: int = Field(default=1000, alias="MAX_CACHE_SIZE")

    # サポートされている取引所
    supported_exchanges: List[str] = Field(default=["bybit"])

    # サポートされているシンボル（Bybit形式）
    supported_symbols: List[str] = Field(default=["BTC/USDT:USDT"])

    # サポートされている時間軸
    supported_timeframes: List[str] = Field(default=["15m", "30m", "1h", "4h", "1d"])

    # デフォルト設定
    default_exchange: str = Field(default="bybit")
    default_symbol: str = Field(default="BTC/USDT:USDT")
    default_timeframe: str = Field(default="1h")
    default_limit: int = Field(default=100)

    # 制限値
    min_limit: int = Field(default=1)
    max_limit: int = Field(default=1000)

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
            "BTC/USDT": "BTC/USDT:USDT",
            "BTC/USDT:USDT": "BTC/USDT:USDT",
            "BTCUSDT_PERP": "BTC/USDT:USDT",
        }
    )

    class Config:
        env_prefix = "MARKET_"
        extra = "ignore"


class GAConfig(BaseSettings):
    """遺伝的アルゴリズム設定"""

    fallback_symbol: str = Field(default="BTC/USDT", alias="GA_FALLBACK_SYMBOL")
    fallback_timeframe: str = Field(default="1d", alias="GA_FALLBACK_TIMEFRAME")
    fallback_start_date: str = Field(
        default="2024-01-01", alias="GA_FALLBACK_START_DATE"
    )
    fallback_end_date: str = Field(default="2024-04-09", alias="GA_FALLBACK_END_DATE")
    fallback_initial_capital: float = Field(
        default=100000.0, alias="GA_FALLBACK_INITIAL_CAPITAL"
    )
    fallback_commission_rate: float = Field(
        default=0.001, alias="GA_FALLBACK_COMMISSION_RATE"
    )

    class Config:
        env_prefix = "GA_"
        extra = "ignore"


class MLDataProcessingConfig(BaseSettings):
    """ML データ処理設定"""

    max_ohlcv_rows: int = Field(default=1000000, description="100万行まで")
    max_feature_rows: int = Field(default=1000000, description="100万行まで")
    feature_calculation_timeout: int = Field(default=3600, description="1時間")
    model_training_timeout: int = Field(default=7200, description="2時間")
    model_prediction_timeout: int = Field(default=10)
    memory_warning_threshold: int = Field(default=8000)
    memory_limit_threshold: int = Field(default=10000)
    debug_mode: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    class Config:
        env_prefix = "ML_DATA_PROCESSING_"
        extra = "ignore"


class MLModelConfig(BaseSettings):
    """ML モデル設定"""

    model_save_path: str = Field(default="models/")
    model_file_extension: str = Field(default=".pkl")
    model_name_prefix: str = Field(default="ml_signal_model")
    auto_strategy_model_name: str = Field(default="auto_strategy_ml_model")
    max_model_versions: int = Field(default=10)
    model_retention_days: int = Field(default=30)

    @model_validator(mode="after")
    def create_model_directory(self):
        """初期化後処理：ディレクトリ作成"""
        os.makedirs(self.model_save_path, exist_ok=True)
        return self

    class Config:
        env_prefix = "ML_MODEL_"
        extra = "ignore"


class MLPredictionConfig(BaseSettings):
    """ML 予測設定"""

    default_up_prob: float = Field(default=0.33)
    default_down_prob: float = Field(default=0.33)
    default_range_prob: float = Field(default=0.34)
    fallback_up_prob: float = Field(default=0.33)
    fallback_down_prob: float = Field(default=0.33)
    fallback_range_prob: float = Field(default=0.34)
    min_probability: float = Field(default=0.0)
    max_probability: float = Field(default=1.0)
    probability_sum_min: float = Field(default=0.8)
    probability_sum_max: float = Field(default=1.2)
    expand_to_data_length: bool = Field(default=True)
    default_indicator_length: int = Field(default=100)

    def get_default_predictions(self) -> Dict[str, float]:
        """デフォルトの予測値を取得"""
        return {
            "up": self.default_up_prob,
            "down": self.default_down_prob,
            "range": self.default_range_prob,
        }

    def get_fallback_predictions(self) -> Dict[str, float]:
        """フォールバック予測値を取得"""
        return {
            "up": self.fallback_up_prob,
            "down": self.fallback_down_prob,
            "range": self.fallback_range_prob,
        }

    class Config:
        env_prefix = "ML_PREDICTION_"
        extra = "ignore"


class MLConfig(BaseSettings):
    """ML 統一設定クラス"""

    data_processing: MLDataProcessingConfig = Field(
        default_factory=MLDataProcessingConfig
    )
    model: MLModelConfig = Field(default_factory=MLModelConfig)
    prediction: MLPredictionConfig = Field(default_factory=MLPredictionConfig)

    class Config:
        env_prefix = "ML_"
        extra = "ignore"


class UnifiedConfig(BaseSettings):
    """
    アプリケーション全体の統一設定クラス

    全ての設定を階層的に管理し、環境変数からの設定読み込みをサポートします。
    """

    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    ga: GAConfig = Field(default_factory=GAConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    def validate_all(self) -> bool:
        """全設定の妥当性を検証"""
        try:
            # 基本設定の検証
            assert self.app.port > 0
            assert self.database.port > 0

            # 市場設定の検証
            assert self.market.min_limit <= self.market.max_limit
            assert self.market.default_limit >= self.market.min_limit
            assert self.market.default_limit <= self.market.max_limit

            # ML設定の検証
            assert self.ml.data_processing.max_ohlcv_rows > 0
            assert self.ml.data_processing.feature_calculation_timeout > 0

            return True
        except AssertionError:
            return False

    class Config:
        env_nested_delimiter = "__"  # ネストされた環境変数をサポート
        extra = "ignore"


# 統一設定のシングルトンインスタンス
unified_config = UnifiedConfig()

# 後方互換性のためのエイリアス
settings = unified_config
