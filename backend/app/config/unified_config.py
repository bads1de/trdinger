"""
統一設定システム

アプリケーション全体の設定を階層的で管理しやすい構造で統一管理します。
SOLID原則に従い、各設定カテゴリを明確に分離し、責任を明確化します。
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.label_generation.enums import ThresholdMethod
from app.utils.label_generation.presets import get_common_presets


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

    model_config = SettingsConfigDict(env_prefix="MARKET_", extra="ignore")


class DataCollectionConfig(BaseSettings):
    """データ収集設定。

    市場データ収集のAPI制限、タイムアウト、メモリ管理などの設定を管理します。
    """

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

    model_config = SettingsConfigDict(env_prefix="DATA_COLLECTION_", extra="ignore")


class BacktestConfig(BaseSettings):
    """バックテスト設定。

    バックテストの初期資金、手数料、結果取得制限などの設定を管理します。
    """

    # デフォルトパラメータ
    default_initial_capital: float = Field(default=10000.0, description="初期資金")
    default_commission_rate: float = Field(default=0.001, description="手数料率")

    # バックテスト実行設定
    max_results_limit: int = Field(default=50, description="結果取得最大件数")
    default_results_limit: int = Field(default=20, description="デフォルト結果件数")

    model_config = SettingsConfigDict(env_prefix="BACKTEST_", extra="ignore")


class AutoStrategyConfig(BaseSettings):
    """自動戦略生成設定。

    遺伝的アルゴリズムによる自動戦略生成の各種パラメータを設定します。
    """

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

    model_config = SettingsConfigDict(env_prefix="AUTO_STRATEGY_", extra="ignore")


class GAConfig(BaseSettings):
    """遺伝的アルゴリズム設定。

    遺伝的アルゴリズムの基本パラメータとバックテスト設定を管理します。
    """

    # 基本設定
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

    # GAパラメータ
    population_size: int = Field(default=50, alias="GA_POPULATION_SIZE")
    generations: int = Field(default=20, alias="GA_GENERATIONS")
    crossover_rate: float = Field(default=0.8, alias="GA_CROSSOVER_RATE")
    mutation_rate: float = Field(default=0.1, alias="GA_MUTATION_RATE")
    elite_size: int = Field(default=5, alias="GA_ELITE_SIZE")

    # 多目的最適化
    enable_multi_objective: bool = Field(
        default=False, alias="GA_ENABLE_MULTI_OBJECTIVE"
    )
    objectives: List[str] = Field(default=["total_return"], alias="GA_OBJECTIVES")
    objective_weights: List[float] = Field(default=[1.0], alias="GA_OBJECTIVE_WEIGHTS")

    # その他設定
    max_indicators: int = Field(default=5, alias="GA_MAX_INDICATORS")
    enable_fitness_sharing: bool = Field(
        default=False, alias="GA_ENABLE_FITNESS_SHARING"
    )
    sharing_radius: float = Field(default=0.1, alias="GA_SHARING_RADIUS")
    sharing_alpha: float = Field(default=1.0, alias="GA_SHARING_ALPHA")

    model_config = SettingsConfigDict(env_prefix="GA_", extra="ignore")


class MLDataProcessingConfig(BaseSettings):
    """ML データ処理設定。

    MLモデルの学習と予測で使用するデータ処理パラメータを設定します。
    """

    max_ohlcv_rows: int = Field(default=1000000, description="100万行まで")
    max_feature_rows: int = Field(default=1000000, description="100万行まで")
    feature_calculation_timeout: int = Field(default=3600, description="1時間")
    model_training_timeout: int = Field(default=7200, description="2時間")
    model_prediction_timeout: int = Field(default=10)
    memory_warning_threshold: int = Field(default=8000)
    memory_limit_threshold: int = Field(default=10000)
    debug_mode: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(env_prefix="ML_DATA_PROCESSING_", extra="ignore")


class MLModelConfig(BaseSettings):
    """ML モデル設定。

    MLモデルの保存パスやバージョン管理などの設定を管理します。
    """

    model_save_path: str = Field(default="models/")
    model_file_extension: str = Field(default=".pkl")
    model_name_prefix: str = Field(default="ml_signal_model")
    auto_strategy_model_name: str = Field(default="auto_strategy_ml_model")
    max_model_versions: int = Field(default=10)
    model_retention_days: int = Field(default=30)

    model_config = SettingsConfigDict(env_prefix="ML_MODEL_", extra="ignore")


class MLPredictionConfig(BaseSettings):
    """ML 予測設定。

    MLモデルの予測結果の確率値やデフォルト値を設定します。
    """

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
        """デフォルトの予測値を取得します。

        Returns:
            Dict[str, float]: 上昇、下降、レンジのデフォルト確率値。
        """
        return {
            "up": self.default_up_prob,
            "down": self.default_down_prob,
            "range": self.default_range_prob,
        }

    def get_fallback_predictions(self) -> Dict[str, float]:
        """フォールバック予測値を取得します。

        Returns:
            Dict[str, float]: 上昇、下降、レンジのフォールバック確率値。
        """
        return {
            "up": self.fallback_up_prob,
            "down": self.fallback_down_prob,
            "range": self.fallback_range_prob,
        }

    model_config = SettingsConfigDict(env_prefix="ML_PREDICTION_", extra="ignore")


@dataclass
class LabelGenerationConfig:
    """ラベル生成の設定。

    機械学習モデルの学習に使用するラベル（目的変数）の生成方法を設定します。
    プリセットを使用するか、カスタム設定を使用するかを選択できます。

    環境変数の例:
        ML__LABEL_GENERATION__DEFAULT_PRESET="4h_4bars"
        ML__LABEL_GENERATION__USE_PRESET=true
        ML__LABEL_GENERATION__TIMEFRAME="4h"
        ML__LABEL_GENERATION__HORIZON_N=4
        ML__LABEL_GENERATION__THRESHOLD=0.002
        ML__LABEL_GENERATION__THRESHOLD_METHOD="FIXED"

    Attributes:
        default_preset: デフォルトのラベル生成プリセット名。
            get_common_presets()で定義されているキーを指定します。
            例: "4h_4bars", "1h_4bars_dynamic"
        timeframe: 時間足（カスタム設定時に使用）。
            サポートされている値: "15m", "30m", "1h", "4h", "1d"
        horizon_n: N本先を見る（カスタム設定時に使用）。
            例: 4本先を見る場合は4を指定
        threshold: 閾値（カスタム設定時に使用）。
            例: 0.002 = 0.2%
        price_column: 価格カラム名（カスタム設定時に使用）。
            通常は"close"を使用
        threshold_method: 閾値計算方法（カスタム設定時に使用）。
            ThresholdMethodのenum値文字列を指定します。
            例: "FIXED", "STD_DEVIATION", "QUANTILE", "KBINS_DISCRETIZER"
        use_preset: プリセットを使うか（True）、カスタム設定を使うか（False）。
    """

    default_preset: str = "4h_4bars"
    timeframe: str = "4h"
    horizon_n: int = 4
    threshold: float = 0.002
    price_column: str = "close"
    threshold_method: str = "FIXED"
    use_preset: bool = True

    def __post_init__(self) -> None:
        """初期化後のバリデーション。"""
        # timeframeの検証
        valid_timeframes = ["15m", "30m", "1h", "4h", "1d"]
        if self.timeframe not in valid_timeframes:
            raise ValueError(
                f"無効な時間足です: {self.timeframe}. "
                f"サポートされている時間足: {', '.join(valid_timeframes)}"
            )

        # threshold_methodの検証
        valid_methods = [method.name for method in ThresholdMethod]
        if self.threshold_method not in valid_methods:
            raise ValueError(
                f"無効な閾値計算方法です: {self.threshold_method}. "
                f"サポートされている方法: {', '.join(valid_methods)}"
            )

        # use_presetがTrueの場合、プリセットの存在確認
        if self.use_preset:
            available_presets = get_common_presets()
            if self.default_preset not in available_presets:
                raise ValueError(
                    f"プリセット '{self.default_preset}' が見つかりません。"
                    f"利用可能なプリセット: {', '.join(sorted(available_presets.keys()))}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換。

        Returns:
            Dict[str, Any]: 設定内容を辞書形式で返します。
        """
        return asdict(self)

    def get_threshold_method_enum(self) -> ThresholdMethod:
        """ThresholdMethod enumを取得。

        Returns:
            ThresholdMethod: 閾値計算方法のenum値。
        """
        return ThresholdMethod[self.threshold_method]


class MLTrainingConfig(BaseSettings):
    """ML 学習アルゴリズム設定。

    各種MLアルゴリズムのハイパーパラメータと学習設定を管理します。
    """

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

    # 一般的な学習設定
    cv_folds: int = Field(default=5, description="クロスバリデーション分割数")
    random_state: int = Field(default=42, description="ランダムシード")

    # ラベル生成設定
    label_generation: LabelGenerationConfig = Field(
        default_factory=LabelGenerationConfig,
        description="ラベル生成の設定",
    )

    @field_validator("label_generation", mode="before")
    @classmethod
    def validate_label_generation(cls, v: Any) -> LabelGenerationConfig:
        """ラベル生成設定のバリデーション。

        環境変数から辞書形式で渡された場合もLabelGenerationConfigに変換します。

        Args:
            v: ラベル生成設定の値

        Returns:
            LabelGenerationConfig: 検証済みのラベル生成設定

        Raises:
            ValueError: バリデーションエラーの場合
        """
        if isinstance(v, LabelGenerationConfig):
            return v
        if isinstance(v, dict):
            return LabelGenerationConfig(**v)
        if v is None:
            return LabelGenerationConfig()
        raise ValueError(f"無効なlabel_generation設定: {type(v)}")

    model_config = SettingsConfigDict(env_prefix="ML_TRAINING_", extra="ignore")


class FeatureEngineeringConfig(BaseSettings):
    """特徴量エンジニアリング設定。
    
    研究目的専用のため、プロファイル機能は簡素化されています。
    """

    # プロファイル機能を削除し、allowlistのみで管理
    # 2025-11-12: 特徴量重要度分析により19個の削除推奨特徴量を削除（79個→60個）
    # 高相関による削除(5個): macd, Stochastic_K, Near_Resistance, MA_Long, BB_Position
    # 低重要度による削除(14個): close_lag_24, cumulative_returns_24, Close_mean_20,
    #   Local_Max, Aroon_Up, BB_Lower, Resistance_Level, BB_Middle,
    #   stochastic_k, rsi_14, bb_lower_20, bb_upper_20, stochastic_d, Local_Min
    feature_allowlist: Optional[List[str]] = Field(
        default=None,
        description="使用する特徴量のリスト（Noneの場合は全特徴量を使用）",
    )

    model_config = SettingsConfigDict(
        env_prefix="ML_FEATURE_ENGINEERING_", extra="ignore"
    )


class MLConfig(BaseSettings):
    """ML 統一設定クラス。

    ML関連のすべての設定を統合的に管理します。
    """

    data_processing: MLDataProcessingConfig = Field(
        default_factory=MLDataProcessingConfig
    )
    model: MLModelConfig = Field(default_factory=MLModelConfig)
    prediction: MLPredictionConfig = Field(default_factory=MLPredictionConfig)
    training: MLTrainingConfig = Field(default_factory=MLTrainingConfig)
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig
    )

    model_config = SettingsConfigDict(env_prefix="ML_", extra="ignore")


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
    ga: GAConfig = Field(default_factory=GAConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__", extra="ignore"  # ネストされた環境変数をサポート
    )


# 統一設定のシングルトンインスタンス
unified_config = UnifiedConfig()
