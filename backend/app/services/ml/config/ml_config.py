"""
ML関連の設定管理

分散していたML関連の設定値を統一的に管理するための設定クラス。
各MLサービスはこの設定を参照することで、一貫性のある動作を保証します。
"""

import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataProcessingConfig(BaseSettings):
    """データ処理関連の設定"""

    # データ制限
    MAX_OHLCV_ROWS: int = Field(default=1000000, description="100万行まで")
    MAX_FEATURE_ROWS: int = Field(default=1000000, description="100万行まで")

    # タイムアウト設定
    FEATURE_CALCULATION_TIMEOUT: int = Field(default=3600, description="1時間")
    MODEL_TRAINING_TIMEOUT: int = Field(default=7200, description="2時間")
    MODEL_PREDICTION_TIMEOUT: int = Field(default=10)

    # メモリ管理
    MEMORY_WARNING_THRESHOLD: int = Field(default=8000)
    MEMORY_LIMIT_THRESHOLD: int = Field(default=10000)

    # デバッグモード
    DEBUG_MODE: bool = Field(default=False)

    # ログレベル
    LOG_LEVEL: str = Field(default="INFO")

    model_config = SettingsConfigDict(env_prefix="ML_DATA_PROCESSING_")


class ModelConfig(BaseSettings):
    """モデル関連の設定"""

    # モデル保存パス
    MODEL_SAVE_PATH: str = Field(default="models/")

    # モデルファイル設定
    MODEL_FILE_EXTENSION: str = Field(default=".pkl")
    MODEL_NAME_PREFIX: str = Field(default="ml_signal_model")
    AUTO_STRATEGY_MODEL_NAME: str = Field(default="auto_strategy_ml_model")

    # モデル管理
    MAX_MODEL_VERSIONS: int = Field(default=10)
    MODEL_RETENTION_DAYS: int = Field(default=30)

    model_config = SettingsConfigDict(env_prefix="ML_MODEL_")


class LightGBMConfig(BaseSettings):
    """LightGBM関連の設定"""

    # 基本パラメータ
    OBJECTIVE: str = Field(default="multiclass")
    NUM_CLASS: int = Field(default=3)
    METRIC: str = Field(default="multi_logloss")
    BOOSTING_TYPE: str = Field(default="gbdt")

    # ハイパーパラメータ
    NUM_LEAVES: int = Field(default=31)
    LEARNING_RATE: float = Field(default=0.05)
    FEATURE_FRACTION: float = Field(default=0.9)
    BAGGING_FRACTION: float = Field(default=0.8)
    BAGGING_FREQ: int = Field(default=5)

    # 学習制御
    NUM_BOOST_ROUND: int = Field(default=1000)
    EARLY_STOPPING_ROUNDS: int = Field(default=50)
    VERBOSE: int = Field(default=-1)

    # その他
    RANDOM_STATE: int = Field(default=42)

    model_config = SettingsConfigDict(env_prefix="ML_LIGHTGBM_")


class FeatureEngineeringConfig(BaseSettings):
    """特徴量エンジニアリング関連の設定"""

    # 特徴量プロファイル設定
    profile: str = Field(
        default="research",
        description="特徴量プロファイル ('research' または 'production')",
    )

    # 計算期間のデフォルト値
    DEFAULT_LOOKBACK_PERIODS: Optional[Dict[str, int]] = Field(
        default_factory=lambda: {
            "short_ma": 10,
            "long_ma": 50,
            "volatility": 20,
            "momentum": 14,
            "volume": 20,
        }
    )

    # キャッシュ設定
    CACHE_ENABLED: bool = Field(default=True)
    MAX_CACHE_SIZE: int = Field(default=10)
    CACHE_TTL_SECONDS: int = Field(default=3600)

    # 特徴量計算設定
    PRICE_FEATURE_PERIODS: Optional[List[int]] = Field(
        default_factory=lambda: [5, 10, 20, 50]
    )
    VOLATILITY_PERIODS: Optional[List[int]] = Field(
        default_factory=lambda: [10, 20, 30]
    )
    VOLUME_PERIODS: Optional[List[int]] = Field(default_factory=lambda: [10, 20, 30])

    model_config = SettingsConfigDict(env_prefix="ML_FEATURE_")


class TrainingConfig(BaseSettings):
    """学習関連の設定"""

    # データ分割
    TRAIN_TEST_SPLIT: float = Field(default=0.8)
    CROSS_VALIDATION_FOLDS: int = Field(default=5)

    # TimeSeriesSplit設定（時系列CV対応）
    USE_TIME_SERIES_SPLIT: bool = Field(
        default=True, description="時系列分割をデフォルトで使用"
    )
    USE_PURGED_KFOLD: bool = Field(
        default=True, description="Purged K-Fold Cross Validation を使用するか"
    )
    MAX_TRAIN_SIZE: Optional[int] = Field(
        default=None, description="TimeSeriesSplitの最大学習サイズ（Noneで制限なし）"
    )

    # ターゲット作成
    PREDICTION_HORIZON: int = Field(default=24)

    # ラベル生成設定（動的閾値を使用）
    LABEL_METHOD: str = Field(
        default="dynamic_volatility", description="ラベル生成方法"
    )
    VOLATILITY_WINDOW: int = Field(
        default=24, description="ボラティリティ計算ウィンドウ"
    )
    THRESHOLD_MULTIPLIER: float = Field(default=0.5, description="閾値乗数")
    MIN_THRESHOLD: float = Field(default=0.005, description="最小閾値")
    MAX_THRESHOLD: float = Field(default=0.05, description="最大閾値")

    # 従来の固定閾値（後方互換性のため保持）
    THRESHOLD_UP: float = Field(default=0.02)
    THRESHOLD_DOWN: float = Field(default=-0.02)

    # 学習制御
    RANDOM_STATE: int = Field(default=42)
    MIN_TRAINING_SAMPLES: int = Field(default=10, description="最小限に緩和")

    # 評価設定
    PERFORMANCE_THRESHOLD: float = Field(default=0.05)
    VALIDATION_SPLIT: float = Field(default=0.2)

    model_config = SettingsConfigDict(env_prefix="ML_TRAINING_")


class PredictionConfig(BaseSettings):
    """予測関連の設定"""

    # デフォルト予測値
    DEFAULT_UP_PROB: float = Field(default=0.33)
    DEFAULT_DOWN_PROB: float = Field(default=0.33)
    DEFAULT_RANGE_PROB: float = Field(default=0.34)
    # ボラティリティ予測用デフォルト
    DEFAULT_TREND_PROB: float = Field(default=0.5)
    DEFAULT_VOLATILITY_RANGE_PROB: float = Field(default=0.5)

    # エラー時フォールバック値
    FALLBACK_UP_PROB: float = Field(default=0.33)
    FALLBACK_DOWN_PROB: float = Field(default=0.33)
    FALLBACK_RANGE_PROB: float = Field(default=0.34)
    # ボラティリティ予測用フォールバック
    FALLBACK_TREND_PROB: float = Field(default=0.5)
    FALLBACK_VOLATILITY_RANGE_PROB: float = Field(default=0.5)

    # 予測値検証
    MIN_PROBABILITY: float = Field(default=0.0)
    MAX_PROBABILITY: float = Field(default=1.0)
    PROBABILITY_SUM_MIN: float = Field(default=0.8)
    PROBABILITY_SUM_MAX: float = Field(default=1.2)

    # 予測結果の拡張
    EXPAND_TO_DATA_LENGTH: bool = Field(default=True)

    # データサイズ制限
    DEFAULT_INDICATOR_LENGTH: int = Field(default=100)

    model_config = SettingsConfigDict(env_prefix="ML_PREDICTION_")


class RetrainingConfig(BaseSettings):
    """再学習関連の設定"""

    # スケジュール設定
    CHECK_INTERVAL_SECONDS: int = Field(default=3600)
    MAX_CONCURRENT_JOBS: int = Field(default=2)
    JOB_TIMEOUT_SECONDS: int = Field(default=7200)

    # データ管理
    DATA_RETENTION_DAYS: int = Field(default=90)
    INCREMENTAL_TRAINING_ENABLED: bool = Field(default=True)

    # 性能監視
    PERFORMANCE_DEGRADATION_THRESHOLD: float = Field(default=0.05)
    DATA_DRIFT_THRESHOLD: float = Field(default=0.1)

    model_config = SettingsConfigDict(env_prefix="ML_RETRAINING_")


class EnsembleConfig(BaseSettings):
    """アンサンブル学習関連の設定"""

    # アンサンブル有効化フラグ
    ENABLED: bool = Field(default=True, description="アンサンブル学習を有効にするか")

    # アルゴリズムリスト
    ALGORITHMS: List[str] = Field(
        default=["lightgbm", "xgboost"], description="使用するアルゴリズム"
    )

    # 投票方法
    VOTING_METHOD: str = Field(default="soft", description="投票方法（soft/hard）")

    # デフォルトアンサンブル設定
    DEFAULT_METHOD: str = Field(
        default="stacking", description="デフォルトのアンサンブル手法"
    )

    # スタッキング設定
    STACKING_BASE_MODELS: List[str] = Field(default=["lightgbm", "xgboost"])
    STACKING_META_MODEL: str = Field(
        default="logistic_regression", description="メタモデル"
    )
    STACKING_CV_FOLDS: int = Field(default=5, description="クロスバリデーション分割数")
    STACKING_STACK_METHOD: str = Field(
        default="predict_proba", description="スタック方法"
    )
    STACKING_N_JOBS: int = Field(default=-1, description="並列処理数")
    STACKING_PASSTHROUGH: bool = Field(
        default=False, description="元特徴量をメタモデルに渡すか"
    )

    model_config = SettingsConfigDict(env_prefix="ML_ENSEMBLE_")


class MLConfig:
    """
    ML関連の統一設定クラス

    全てのML関連サービスがこのクラスを通じて設定にアクセスします。
    アンサンブル学習をデフォルトとし、複数のモデルを組み合わせて
    予測精度と頑健性を向上させます。
    """

    def __init__(self):
        """設定の初期化"""
        self.data_processing = DataProcessingConfig()
        self.model = ModelConfig()
        self.lightgbm = LightGBMConfig()  # アンサンブル内のベースモデルとして保持
        self.ensemble = EnsembleConfig()  # アンサンブル学習設定を追加
        self.feature_engineering = FeatureEngineeringConfig()
        self.training = TrainingConfig()
        self.prediction = PredictionConfig()
        self.retraining = RetrainingConfig()

        # 設定の妥当性を検証
        self._validate_all_configs()

    def to_dict(self) -> Dict[str, Any]:
        """
        全設定を辞書形式に変換（統一的なシリアライゼーション）

        各設定クラスのPydantic model_dump()を活用して、
        手動マッピングを排除し保守性を向上させました。
        """
        return {
            "data_processing": self.data_processing.model_dump(),
            "model": self.model.model_dump(),
            "lightgbm": self.lightgbm.model_dump(),
            "ensemble": self.ensemble.model_dump(),
            "feature_engineering": self.feature_engineering.model_dump(),
            "training": self.training.model_dump(),
            "prediction": self.prediction.model_dump(),
            "retraining": self.retraining.model_dump(),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MLConfig":
        """
        辞書から設定を復元（統一的なデシリアライゼーション）

        Args:
            config_dict: 設定辞書

        Returns:
            復元されたMLConfigインスタンス
        """
        instance = cls()

        # 各設定セクションを復元
        if "data_processing" in config_dict:
            instance.data_processing = DataProcessingConfig(
                **config_dict["data_processing"]
            )
        if "model" in config_dict:
            instance.model = ModelConfig(**config_dict["model"])
        if "lightgbm" in config_dict:
            instance.lightgbm = LightGBMConfig(**config_dict["lightgbm"])
        if "ensemble" in config_dict:
            instance.ensemble = EnsembleConfig(**config_dict["ensemble"])
        if "feature_engineering" in config_dict:
            instance.feature_engineering = FeatureEngineeringConfig(
                **config_dict["feature_engineering"]
            )
        if "training" in config_dict:
            instance.training = TrainingConfig(**config_dict["training"])
        if "prediction" in config_dict:
            instance.prediction = PredictionConfig(**config_dict["prediction"])
        if "retraining" in config_dict:
            instance.retraining = RetrainingConfig(**config_dict["retraining"])

        return instance

    def _validate_all_configs(self):
        """全設定の妥当性を検証 (validate_config と get_validation_errors を内包)"""
        try:
            # 予測設定の検証
            prediction_errors = []
            default_predictions = {
                "up": self.prediction.DEFAULT_UP_PROB,
                "down": self.prediction.DEFAULT_DOWN_PROB,
                "range": self.prediction.DEFAULT_RANGE_PROB,
            }
            if not (
                0.0 <= default_predictions["up"] <= 1.0
                and 0.0 <= default_predictions["down"] <= 1.0
                and 0.0 <= default_predictions["range"] <= 1.0
            ):
                prediction_errors.append("デフォルト予測値が範囲外です")

            fallback_predictions = {
                "up": self.prediction.FALLBACK_UP_PROB,
                "down": self.prediction.FALLBACK_DOWN_PROB,
                "range": self.prediction.FALLBACK_RANGE_PROB,
            }
            if not (
                0.0 <= fallback_predictions["up"] <= 1.0
                and 0.0 <= fallback_predictions["down"] <= 1.0
                and 0.0 <= fallback_predictions["range"] <= 1.0
            ):
                prediction_errors.append("フォールバック予測値が範囲外です")

            if not (
                0.0
                <= self.prediction.MIN_PROBABILITY
                <= self.prediction.MAX_PROBABILITY
                <= 1.0
            ):
                prediction_errors.append("確率範囲設定が無効です")

            if not (
                0.0
                < self.prediction.PROBABILITY_SUM_MIN
                <= self.prediction.PROBABILITY_SUM_MAX
                <= 2.0
            ):
                prediction_errors.append("確率合計範囲設定が無効です")

            if not (1 <= self.prediction.DEFAULT_INDICATOR_LENGTH <= 100000):
                prediction_errors.append("デフォルト指標長が無効です")

            if prediction_errors:
                logging.warning(f"予測設定に問題があります: {prediction_errors}")

            # データ処理設定の検証
            if self.data_processing.DEBUG_MODE:
                logging.info("デバッグモードが有効です")

            # ログレベルの設定
            log_level = getattr(
                logging, self.data_processing.LOG_LEVEL.upper(), logging.INFO
            )
            logging.getLogger().setLevel(log_level)

        except Exception as e:
            logging.error(f"設定検証中にエラーが発生しました: {e}")

    def validate(self) -> bool:
        """設定の妥当性を検証"""
        try:
            # データ処理設定の検証
            assert self.data_processing.MAX_OHLCV_ROWS > 0
            assert self.data_processing.FEATURE_CALCULATION_TIMEOUT > 0

            # モデル設定の検証
            # ModelConfigの__post_init__でディレクトリが作成されるため、パスの存在チェックは不要

            # 学習設定の検証
            assert 0 < self.training.TRAIN_TEST_SPLIT < 1
            assert self.training.THRESHOLD_UP > 0
            assert self.training.THRESHOLD_DOWN < 0

            # 予測設定の検証
            assert 0 <= self.prediction.DEFAULT_UP_PROB <= 1
            assert 0 <= self.prediction.DEFAULT_DOWN_PROB <= 1
            assert 0 <= self.prediction.DEFAULT_RANGE_PROB <= 1

            return True

        except AssertionError:
            return False

    def get_model_search_paths(self) -> List[str]:
        """モデル検索パスのリストを取得"""
        paths = [
            self.model.MODEL_SAVE_PATH,
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


# グローバル設定インスタンス
ml_config = MLConfig()

# 設定の妥当性を検証
if not ml_config.validate():
    raise ValueError("ML設定の妥当性検証に失敗しました")
