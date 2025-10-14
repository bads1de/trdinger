"""
統一設定システム

アプリケーション全体の設定を階層的で管理しやすい構造で統一管理します。
SOLID原則に従い、各設定カテゴリを明確に分離し、責任を明確化します。
"""

from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


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
            "BTCUSDT_PERP": "BTC/USDT:USDT",
        }
    )

    class Config:
        env_prefix = "MARKET_"
        extra = "ignore"


class DataCollectionConfig(BaseSettings):
    """データ収集設定"""

    # API制限設定
    default_limit: int = Field(default=100, description="デフォルト取得件数")
    max_limit: int = Field(default=1000, description="最大取得件数")
    min_limit: int = Field(default=1, description="最小取得件数")

    # Bybit API設定
    bybit_timeout: int = Field(default=30, description="APIタイムアウト（秒）")
    bybit_page_limit: int = Field(default=200, description="ページング時の制限")
    bybit_max_pages: int = Field(
        default=500, description="最大ページ数（2020年からのデータを取得可能に拡張）"
    )

    # メモリ管理
    memory_warning_threshold: int = Field(default=8000, description="メモリ警告閾値")
    memory_limit_threshold: int = Field(default=10000, description="メモリ制限閾値")

    class Config:
        env_prefix = "DATA_COLLECTION_"
        extra = "ignore"


class BacktestConfig(BaseSettings):
    """バックテスト設定"""

    # デフォルトパラメータ
    default_initial_capital: float = Field(default=10000.0, description="初期資金")
    default_commission_rate: float = Field(default=0.001, description="手数料率")

    # バックテスト実行設定
    max_results_limit: int = Field(default=50, description="結果取得最大件数")
    default_results_limit: int = Field(default=20, description="デフォルト結果件数")

    class Config:
        env_prefix = "BACKTEST_"
        extra = "ignore"


class AutoStrategyConfig(BaseSettings):
    """自動戦略生成設定"""

    # 遺伝的アルゴリズム設定
    population_size: int = Field(default=50, description="個体数")
    generations: int = Field(default=20, description="世代数")
    tournament_size: int = Field(default=3, description="トーナメントサイズ")
    mutation_rate: float = Field(default=0.1, description="突然変異率")

    # 戦略生成設定
    max_indicators: int = Field(default=5, description="最大指標数")
    min_indicators: int = Field(default=2, description="最小指標数")
    max_conditions: int = Field(default=5, description="最大条件数")
    min_conditions: int = Field(default=2, description="最小条件数")

    # フィットネス共有設定
    fitness_sharing_radius: float = Field(default=0.1, description="共有半径")

    # 戦略API設定
    default_strategies_limit: int = Field(
        default=20, description="戦略取得デフォルト件数"
    )
    max_strategies_limit: int = Field(default=100, description="戦略取得最大件数")

    # ボラティリティTP/SL設定
    atr_period: int = Field(default=14, description="ATR計算期間")
    atr_multiplier_sl: float = Field(default=2.0, description="SL用ATR倍率")
    atr_multiplier_tp: float = Field(default=3.0, description="TP用ATR倍率")
    min_sl_pct: float = Field(default=0.005, description="最小SLパーセンテージ")
    max_sl_pct: float = Field(default=0.1, description="最大SLパーセンテージ")
    min_tp_pct: float = Field(default=0.01, description="最小TPパーセンテージ")
    max_tp_pct: float = Field(default=0.2, description="最大TPパーセンテージ")
    regime_lookback: int = Field(
        default=50, description="ボラティリティレジーム判定期間"
    )
    estimated_atr_pct: float = Field(default=0.02, description="推定ATRパーセンテージ")
    default_atr_mean: float = Field(default=0.02, description="デフォルトATR平均")
    default_atr_std: float = Field(default=0.01, description="デフォルトATR標準偏差")

    # ポジションサイジング設定
    default_atr_multiplier: float = Field(default=0.02, description="デフォルトATR倍率")
    fallback_atr_multiplier: float = Field(
        default=0.04, description="フォールバックATR倍率"
    )
    assumed_win_rate: float = Field(default=0.55, description="想定勝率")
    assumed_avg_win: float = Field(default=0.02, description="想定平均勝ち額")
    assumed_avg_loss: float = Field(default=0.015, description="想定平均負け額")
    default_position_ratio: float = Field(
        default=0.1, description="デフォルトポジション比率"
    )

    # ボラティリティレジーム閾値
    very_low_threshold: float = Field(default=-1.5, description="非常に低い閾値")
    low_threshold: float = Field(default=-0.5, description="低い閾値")
    high_threshold: float = Field(default=1.5, description="高い閾値")

    class Config:
        env_prefix = "AUTO_STRATEGY_"
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


class MLTrainingConfig(BaseSettings):
    """ML 学習アルゴリズム設定"""

    # LightGBM デフォルトパラメータ
    lgb_n_estimators: int = Field(default=100, description="推定器数")
    lgb_learning_rate: float = Field(default=0.1, description="学習率")
    lgb_max_depth: int = Field(default=10, description="最大深度")
    lgb_num_leaves: int = Field(default=31, description="葉の数")

    # XGBoost デフォルトパラメータ
    xgb_n_estimators: int = Field(default=100, description="推定器数")
    xgb_learning_rate: float = Field(default=0.1, description="学習率")
    xgb_max_depth: int = Field(default=6, description="最大深度")

    # RandomForest デフォルトパラメータ
    rf_n_estimators: int = Field(default=100, description="推定器数")
    rf_max_depth: int = Field(default=10, description="最大深度")

    # GradientBoosting デフォルトパラメータ
    gb_n_estimators: int = Field(default=50, description="推定器数")
    gb_learning_rate: float = Field(default=0.2, description="学習率")
    gb_max_depth: int = Field(default=3, description="最大深度")
    gb_subsample: float = Field(default=0.8, description="サブサンプル比率")

    # LogisticRegression デフォルトパラメータ
    lr_max_iter: int = Field(default=1000, description="最大イテレーション数")

    # CatBoost デフォルトパラメータ
    cat_iterations: int = Field(default=1000, description="イテレーション数")
    cat_learning_rate: float = Field(default=0.1, description="学習率")

    # 一般的な学習設定
    cv_folds: int = Field(default=5, description="クロスバリデーション分割数")
    random_state: int = Field(default=42, description="ランダムシード")

    class Config:
        env_prefix = "ML_TRAINING_"
        extra = "ignore"


class MLConfig(BaseSettings):
    """ML 統一設定クラス"""

    data_processing: MLDataProcessingConfig = Field(
        default_factory=MLDataProcessingConfig
    )
    model: MLModelConfig = Field(default_factory=MLModelConfig)
    prediction: MLPredictionConfig = Field(default_factory=MLPredictionConfig)
    training: MLTrainingConfig = Field(default_factory=MLTrainingConfig)

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
    market: MarketConfig = Field(default_factory=MarketConfig)
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    auto_strategy: AutoStrategyConfig = Field(default_factory=AutoStrategyConfig)
    ga: GAConfig = Field(default_factory=GAConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    class Config:
        env_nested_delimiter = "__"  # ネストされた環境変数をサポート
        extra = "ignore"


# 統一設定のシングルトンインスタンス
unified_config = UnifiedConfig()
