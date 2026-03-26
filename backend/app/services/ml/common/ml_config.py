"""
ML関連設定クラス

機械学習パイプラインの設定を一元管理します。
設定はサービス配下で完結し、環境変数ベースで上書きできます。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.utils.serialization import dataclass_to_dict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.constants import DEFAULT_ENSEMBLE_ALGORITHMS, SUPPORTED_TIMEFRAMES


class EnsembleConfig(BaseSettings):
    """アンサンブル学習設定。

    アンサンブル学習のアルゴリズム、投票方法、スタッキング設定などを管理します。
    """

    enabled: bool = Field(
        default=True, alias="ENABLED", description="アンサンブル学習を有効にするか"
    )
    algorithms: List[str] = Field(
        default_factory=lambda: list(DEFAULT_ENSEMBLE_ALGORITHMS),
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
        default_factory=lambda: list(DEFAULT_ENSEMBLE_ALGORITHMS),
        alias="STACKING_BASE_MODELS",
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
        from app.services.ml.label_generation.label_cache import ThresholdMethod
        from app.services.ml.label_generation.presets import get_common_presets

        # timeframeの検証
        valid_timeframes = list(SUPPORTED_TIMEFRAMES)
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
        return dataclass_to_dict(self)


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
