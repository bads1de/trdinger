"""
統一設定システム

アプリケーション全体の設定を階層的で管理しやすい構造で統一管理します。
SOLID原則に従い、各設定カテゴリを明確に分離し、責任を明確化します。
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnsembleConfig(BaseSettings):
    """アンサンブル学習設定。

    アンサンブル学習のアルゴリズム、投票方法、スタッキング設定などを管理します。
    """

    enabled: bool = Field(
        default=True, alias="ENABLED", description="アンサンブル学習を有効にするか"
    )
    algorithms: List[str] = Field(
        default=["lightgbm", "xgboost", "catboost"],
        alias="ALGORITHMS",
        description="使用するアルゴリズム",
    )
    voting_method: str = Field(
        default="soft", alias="VOTING_METHOD", description="投票方法（soft/hard）"
    )
    default_method: str = Field(
        default="stacking",
        alias="DEFAULT_METHOD",
        description="デフォルトのアンサンブル手法",
    )

    # スタッキング設定
    stacking_base_models: List[str] = Field(
        default=["lightgbm", "xgboost", "catboost"], alias="STACKING_BASE_MODELS"
    )
    stacking_meta_model: str = Field(
        default="logistic_regression",
        alias="STACKING_META_MODEL",
        description="メタモデル",
    )
    stacking_cv_folds: int = Field(
        default=5, alias="STACKING_CV_FOLDS", description="クロスバリデーション分割数"
    )
    stacking_stack_method: str = Field(
        default="predict_proba",
        alias="STACKING_STACK_METHOD",
        description="スタック方法",
    )
    stacking_n_jobs: int = Field(
        default=-1, alias="STACKING_N_JOBS", description="並列処理数"
    )
    stacking_passthrough: bool = Field(
        default=False,
        alias="STACKING_PASSTHROUGH",
        description="元特徴量をメタモデルに渡すか",
    )

    model_config = SettingsConfigDict(env_prefix="ML_ENSEMBLE_", extra="ignore")


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
    supported_timeframes: List[str] = Field(
        default=["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )

    # デフォルト設定
    default_exchange: str = Field(default="bybit")
    default_symbol: str = Field(default="BTC/USDT:USDT")
    default_timeframe: str = Field(default="1h")
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

    # 遺伝的アルゴリズム基本設定
    population_size: int = Field(default=50, description="個体数")
    generations: int = Field(default=20, description="世代数")
    tournament_size: int = Field(default=3, description="トーナメントサイズ")
    crossover_rate: float = Field(default=0.8, description="交叉率")
    mutation_rate: float = Field(default=0.1, description="突然変異率")
    elite_size: int = Field(default=5, description="エリート保存数")

    # 戦略生成制約
    max_indicators: int = Field(default=5, description="最大指標数")
    min_indicators: int = Field(default=2, description="最小指標数")
    max_conditions: int = Field(default=5, description="最大条件数")
    min_conditions: int = Field(default=2, description="最小条件数")

    # 多目的最適化設定
    enable_multi_objective: bool = Field(
        default=False, description="多目的最適化を有効にするか"
    )
    objectives: List[str] = Field(
        default=["total_return"], description="最適化対象の指標"
    )
    objective_weights: List[float] = Field(default=[1.0], description="各指標の重み")

    # フィットネス共有設定
    enable_fitness_sharing: bool = Field(
        default=False, description="フィットネス共有を有効にするか"
    )
    fitness_sharing_radius: float = Field(default=0.1, description="共有半径")
    sharing_alpha: float = Field(default=1.0, description="共有アルファ")

    # フォールバック設定（GA実行時のデフォルト値）
    fallback_symbol: str = Field(
        default="BTC/USDT:USDT", description="フォールバックシンボル"
    )
    fallback_timeframe: str = Field(default="1d", description="フォールバック時間足")
    fallback_start_date: str = Field(
        default="2024-01-01", description="フォールバック開始日"
    )
    fallback_end_date: str = Field(
        default="2024-04-09", description="フォールバック終了日"
    )
    fallback_initial_capital: float = Field(
        default=100000.0, description="フォールバック初期資金"
    )
    fallback_commission_rate: float = Field(
        default=0.001, description="フォールバック手数料率"
    )

    # 戦略API設定
    default_strategies_limit: int = Field(
        default=20, description="戦略取得デフォルト件数"
    )
    max_strategies_limit: int = Field(default=100, description="戦略取得最大件数")

    model_config = SettingsConfigDict(env_prefix="AUTO_STRATEGY_", extra="ignore")


class MLDataProcessingConfig(BaseSettings):
    """ML データ処理設定。

    MLモデルの学習と予測で使用するデータ処理パラメータを設定します。
    """

    max_ohlcv_rows: int = Field(
        default=1000000, description="100万行まで", alias="MAX_OHLCV_ROWS"
    )
    max_feature_rows: int = Field(
        default=1000000, description="100万行まで", alias="MAX_FEATURE_ROWS"
    )
    feature_calculation_timeout: int = Field(
        default=3600, description="1時間", alias="FEATURE_CALCULATION_TIMEOUT"
    )
    model_training_timeout: int = Field(
        default=7200, description="2時間", alias="MODEL_TRAINING_TIMEOUT"
    )
    model_prediction_timeout: int = Field(default=10, alias="MODEL_PREDICTION_TIMEOUT")

    model_config = SettingsConfigDict(env_prefix="ML_DATA_PROCESSING_", extra="ignore")


class MLModelConfig(BaseSettings):
    """ML モデル設定。

    MLモデルの保存パスやバージョン管理などの設定を管理します。
    """

    model_save_path: str = Field(default="models/", alias="MODEL_SAVE_PATH")
    model_file_extension: str = Field(default=".pkl", alias="MODEL_FILE_EXTENSION")
    model_name_prefix: str = Field(default="ml_signal_model", alias="MODEL_NAME_PREFIX")
    auto_strategy_model_name: str = Field(
        default="auto_strategy_ml_model", alias="AUTO_STRATEGY_MODEL_NAME"
    )
    max_model_versions: int = Field(default=10, alias="MAX_MODEL_VERSIONS")
    model_retention_days: int = Field(default=30, alias="MODEL_RETENTION_DAYS")

    model_config = SettingsConfigDict(env_prefix="ML_MODEL_", extra="ignore")


class MLPredictionConfig(BaseSettings):
    """ML 予測設定。

    MLモデルの予測結果の確率値やデフォルト値を設定します。
    二値分類（メタラベリング / ダマシ予測）専用です。
    """

    # 二値分類用（is_valid: エントリーが有効である確率）
    default_is_valid_prob: float = Field(
        default=0.5, alias="DEFAULT_IS_VALID_PROB", description="デフォルトの有効確率"
    )
    fallback_is_valid_prob: float = Field(
        default=0.5,
        alias="FALLBACK_IS_VALID_PROB",
        description="フォールバック有効確率",
    )

    min_probability: float = Field(default=0.0, alias="MIN_PROBABILITY")
    max_probability: float = Field(default=1.0, alias="MAX_PROBABILITY")
    expand_to_data_length: bool = Field(default=True, alias="EXPAND_TO_DATA_LENGTH")
    default_indicator_length: int = Field(default=100, alias="DEFAULT_INDICATOR_LENGTH")

    def get_default_predictions(self) -> Dict[str, float]:
        """デフォルトの予測値を取得します。

        Returns:
            Dict[str, float]: is_validのデフォルト確率値。
        """
        return {
            "is_valid": self.default_is_valid_prob,
        }

    def get_fallback_predictions(self) -> Dict[str, float]:
        """フォールバック予測値を取得します。

        Returns:
            Dict[str, float]: is_validのフォールバック確率値。
        """
        return {
            "is_valid": self.fallback_is_valid_prob,
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
            例: "tbm_4h_1.0_1.0"
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
            例: "TREND_SCANNING"
        use_preset: プリセットを使うか（True）、カスタム設定を使うか（False）。
    """

    default_preset: str = "tbm_4h_0.5_1.0"
    timeframe: str = "4h"
    horizon_n: int = 4
    threshold: float = 0.002
    price_column: str = "close"
    threshold_method: str = "TREND_SCANNING"
    use_preset: bool = True

    def __post_init__(self) -> None:
        """初期化後のバリデーション。"""
        # 循環依存を避けるための遅延インポート
        from app.services.ml.label_generation.enums import ThresholdMethod
        from app.services.ml.label_generation.presets import get_common_presets

        # timeframeの検証
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
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


class MLTrainingConfig(BaseSettings):
    """ML 学習アルゴリズム設定。

    各種MLアルゴリズムのハイパーパラメータと学習設定を管理します。
    """

    # LightGBM デフォルトパラメータ
    lgb_n_estimators: int = Field(default=100, description="推定器数")
    lgb_learning_rate: float = Field(default=0.1, description="学習率")
    lgb_max_depth: int = Field(default=10, description="最大深度")
    lgb_num_leaves: int = Field(default=31, description="葉の数")
    lgb_objective: str = Field(default="binary", description="目的関数（二値分類）")
    lgb_num_class: int = Field(default=2, description="クラス数（二値分類）")
    lgb_metric: str = Field(default="binary_logloss", description="評価指標")
    lgb_boosting_type: str = Field(default="gbdt", description="ブースティングタイプ")
    lgb_feature_fraction: float = Field(default=0.9, description="特徴量採用率")
    lgb_bagging_fraction: float = Field(default=0.8, description="バギング採用率")
    lgb_bagging_freq: int = Field(default=5, description="バギング頻度")
    lgb_early_stopping_rounds: int = Field(default=50, description="早期終了ラウンド数")
    lgb_verbose: int = Field(default=-1, description="詳細出力レベル")

    # XGBoost デフォルトパラメータ
    xgb_n_estimators: int = Field(default=100, description="推定器数")
    xgb_learning_rate: float = Field(default=0.1, description="学習率")
    xgb_max_depth: int = Field(default=6, description="最大深度")

    # LogisticRegression デフォルトパラメータ
    lr_max_iter: int = Field(default=1000, description="最大イテレーション数")

    # 一般的な学習設定
    cv_folds: int = Field(
        default=5,
        description="クロスバリデーション分割数",
        alias="CROSS_VALIDATION_FOLDS",
    )
    random_state: int = Field(default=42, description="ランダムシード")

    # データ分割
    train_test_split: float = Field(default=0.8, alias="TRAIN_TEST_SPLIT")

    # PurgedKFold設定
    pct_embargo: float = Field(
        default=0.01,
        description="エンバーゴ率（データ漏洩防止のための待機期間）",
        alias="PCT_EMBARGO",
    )

    # 学習制御
    min_training_samples: int = Field(
        default=10, description="最小限に緩和", alias="MIN_TRAINING_SAMPLES"
    )

    performance_threshold: float = Field(default=0.05, alias="PERFORMANCE_THRESHOLD")
    validation_split: float = Field(default=0.2, alias="VALIDATION_SPLIT")

    # クラス不均衡対策
    use_class_weight: bool = Field(
        default=False, description="class_weightを使用するか"
    )
    class_weight_mode: str = Field(
        default="balanced", description="class_weightモード ('balanced' or custom dict)"
    )
    use_smote: bool = Field(default=False, description="SMOTE/ADASYNを使用するか")
    smote_method: str = Field(
        default="smote", description="サンプリング方法 ('smote' or 'adasyn')"
    )

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
        if hasattr(v, "__class__") and v.__class__.__name__ == "LabelGenerationConfig":
            return v
        if isinstance(v, dict):
            return LabelGenerationConfig(**v)
        if v is None:
            return LabelGenerationConfig()
        raise ValueError(f"無効なlabel_generation設定: {type(v)}")

    model_config = SettingsConfigDict(env_prefix="ML_TRAINING_", extra="ignore")


# --- 機械学習設定 ---


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
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)

    def get_model_search_paths(self) -> List[str]:
        """モデル検索パスのリストを取得"""
        import os

        paths = [
            self.model.model_save_path,
            "backend/models/",
            "models/",
            "backend/ml_models/",
            "ml_models/",
        ]

        # 重複を除去し、存在するパスのみを返す
        unique_paths = []
        seen_absolute_paths = set()

        for path in paths:
            abs_path = os.path.abspath(path)
            if abs_path not in seen_absolute_paths:
                unique_paths.append(path)
                seen_absolute_paths.add(abs_path)

        return unique_paths

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
    ml: MLConfig = Field(default_factory=MLConfig)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",  # ネストされた環境変数をサポート
    )


# 統一設定のシングルトンインスタンス
unified_config = UnifiedConfig()
