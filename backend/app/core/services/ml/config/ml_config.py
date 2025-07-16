"""
ML関連の設定管理

分散していたML関連の設定値を統一的に管理するための設定クラス。
各MLサービスはこの設定を参照することで、一貫性のある動作を保証します。
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class DataProcessingConfig:
    """データ処理関連の設定"""

    # データ制限
    MAX_OHLCV_ROWS: int = 10000
    MAX_FEATURE_ROWS: int = 50000

    # タイムアウト設定
    FEATURE_CALCULATION_TIMEOUT: int = 30
    MODEL_TRAINING_TIMEOUT: int = 300
    MODEL_PREDICTION_TIMEOUT: int = 10

    # メモリ管理
    MEMORY_WARNING_THRESHOLD: int = 8000
    MEMORY_LIMIT_THRESHOLD: int = 10000

    # デバッグモード
    DEBUG_MODE: bool = False

    # ログレベル
    LOG_LEVEL: str = "INFO"


@dataclass
class ModelConfig:
    """モデル関連の設定"""

    # モデル保存パス
    MODEL_SAVE_PATH: str = "models/"
    MODEL_BACKUP_PATH: str = "models/backup/"

    # モデルファイル設定
    MODEL_FILE_EXTENSION: str = ".pkl"
    MODEL_NAME_PREFIX: str = "ml_signal_model"
    AUTO_STRATEGY_MODEL_NAME: str = "auto_strategy_ml_model"

    # モデル管理
    MAX_MODEL_VERSIONS: int = 10
    MODEL_RETENTION_DAYS: int = 30

    def __post_init__(self):
        """初期化後処理：ディレクトリ作成"""
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(self.MODEL_BACKUP_PATH, exist_ok=True)


@dataclass
class LightGBMConfig:
    """LightGBM関連の設定"""

    # 基本パラメータ
    OBJECTIVE: str = "multiclass"
    NUM_CLASS: int = 3
    METRIC: str = "multi_logloss"
    BOOSTING_TYPE: str = "gbdt"

    # ハイパーパラメータ
    NUM_LEAVES: int = 31
    LEARNING_RATE: float = 0.05
    FEATURE_FRACTION: float = 0.9
    BAGGING_FRACTION: float = 0.8
    BAGGING_FREQ: int = 5

    # 学習制御
    NUM_BOOST_ROUND: int = 1000
    EARLY_STOPPING_ROUNDS: int = 50
    VERBOSE: int = -1

    # その他
    RANDOM_STATE: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で設定を返す"""
        return {
            "objective": self.OBJECTIVE,
            "num_class": self.NUM_CLASS,
            "metric": self.METRIC,
            "boosting_type": self.BOOSTING_TYPE,
            "num_leaves": self.NUM_LEAVES,
            "learning_rate": self.LEARNING_RATE,
            "feature_fraction": self.FEATURE_FRACTION,
            "bagging_fraction": self.BAGGING_FRACTION,
            "bagging_freq": self.BAGGING_FREQ,
            "verbose": self.VERBOSE,
            "random_state": self.RANDOM_STATE,
        }


@dataclass
class FeatureEngineeringConfig:
    """特徴量エンジニアリング関連の設定"""

    # 計算期間のデフォルト値
    DEFAULT_LOOKBACK_PERIODS: Optional[Dict[str, int]] = None

    # キャッシュ設定
    CACHE_ENABLED: bool = True
    MAX_CACHE_SIZE: int = 10
    CACHE_TTL_SECONDS: int = 3600

    # 特徴量計算設定
    PRICE_FEATURE_PERIODS: Optional[List[int]] = None
    VOLATILITY_PERIODS: Optional[List[int]] = None
    VOLUME_PERIODS: Optional[List[int]] = None

    def __post_init__(self):
        """デフォルト値の設定"""
        if self.DEFAULT_LOOKBACK_PERIODS is None:
            self.DEFAULT_LOOKBACK_PERIODS = {
                "short_ma": 10,
                "long_ma": 50,
                "volatility": 20,
                "momentum": 14,
                "volume": 20,
            }

        if self.PRICE_FEATURE_PERIODS is None:
            self.PRICE_FEATURE_PERIODS = [5, 10, 20, 50]

        if self.VOLATILITY_PERIODS is None:
            self.VOLATILITY_PERIODS = [10, 20, 30]

        if self.VOLUME_PERIODS is None:
            self.VOLUME_PERIODS = [10, 20, 30]


@dataclass
class TrainingConfig:
    """学習関連の設定"""

    # データ分割
    TRAIN_TEST_SPLIT: float = 0.8
    CROSS_VALIDATION_FOLDS: int = 5

    # ターゲット作成
    PREDICTION_HORIZON: int = 24
    THRESHOLD_UP: float = 0.02
    THRESHOLD_DOWN: float = -0.02

    # 学習制御
    RANDOM_STATE: int = 42
    MIN_TRAINING_SAMPLES: int = 1000

    # 評価設定
    PERFORMANCE_THRESHOLD: float = 0.05
    VALIDATION_SPLIT: float = 0.2


@dataclass
class PredictionConfig:
    """予測関連の設定"""

    # デフォルト予測値
    DEFAULT_UP_PROB: float = 0.33
    DEFAULT_DOWN_PROB: float = 0.33
    DEFAULT_RANGE_PROB: float = 0.34

    # エラー時フォールバック値
    FALLBACK_UP_PROB: float = 0.33
    FALLBACK_DOWN_PROB: float = 0.33
    FALLBACK_RANGE_PROB: float = 0.34

    # 予測値検証
    MIN_PROBABILITY: float = 0.0
    MAX_PROBABILITY: float = 1.0
    PROBABILITY_SUM_MIN: float = 0.8
    PROBABILITY_SUM_MAX: float = 1.2

    # 予測結果の拡張
    EXPAND_TO_DATA_LENGTH: bool = True

    # データサイズ制限
    DEFAULT_INDICATOR_LENGTH: int = 100

    def get_default_predictions(self) -> Dict[str, float]:
        """デフォルトの予測値を取得"""
        return {
            "up": self.DEFAULT_UP_PROB,
            "down": self.DEFAULT_DOWN_PROB,
            "range": self.DEFAULT_RANGE_PROB,
        }

    def get_fallback_predictions(self) -> Dict[str, float]:
        """エラー時のフォールバック予測値を取得"""
        return {
            "up": self.FALLBACK_UP_PROB,
            "down": self.FALLBACK_DOWN_PROB,
            "range": self.FALLBACK_RANGE_PROB,
        }

    def get_default_indicators(self, data_length: int) -> Dict[str, Any]:
        """デフォルトのML指標を取得"""
        import numpy as np

        return {
            "ML_UP_PROB": np.full(data_length, self.DEFAULT_UP_PROB),
            "ML_DOWN_PROB": np.full(data_length, self.DEFAULT_DOWN_PROB),
            "ML_RANGE_PROB": np.full(data_length, self.DEFAULT_RANGE_PROB),
        }

    def validate_predictions(self, predictions: Dict[str, float]) -> bool:
        """予測値の妥当性を検証"""
        try:
            required_keys = ["up", "down", "range"]

            # 必要なキーが存在するか
            if not all(key in predictions for key in required_keys):
                return False

            # 値が数値で0-1の範囲内か
            for key in required_keys:
                value = predictions[key]
                if not isinstance(value, (int, float)):
                    return False
                if not (self.MIN_PROBABILITY <= value <= self.MAX_PROBABILITY):
                    return False

            # 合計が妥当な範囲内か
            total = sum(predictions[key] for key in required_keys)
            if not (self.PROBABILITY_SUM_MIN <= total <= self.PROBABILITY_SUM_MAX):
                return False

            return True

        except Exception:
            return False

    def validate_config(self) -> bool:
        """設定値の妥当性を検証"""
        try:
            # デフォルト値の検証
            default_predictions = self.get_default_predictions()
            if not self.validate_predictions(default_predictions):
                return False

            # フォールバック値の検証
            fallback_predictions = self.get_fallback_predictions()
            if not self.validate_predictions(fallback_predictions):
                return False

            # 範囲値の検証
            if not (0.0 <= self.MIN_PROBABILITY <= self.MAX_PROBABILITY <= 1.0):
                return False

            if not (0.0 < self.PROBABILITY_SUM_MIN <= self.PROBABILITY_SUM_MAX <= 2.0):
                return False

            # データ長の検証
            if not (1 <= self.DEFAULT_INDICATOR_LENGTH <= 100000):
                return False

            return True

        except Exception:
            return False

    def get_validation_errors(self) -> List[str]:
        """設定値の検証エラーを取得"""
        errors = []

        try:
            # デフォルト値の検証
            default_predictions = self.get_default_predictions()
            if not self.validate_predictions(default_predictions):
                errors.append("デフォルト予測値が無効です")

            # フォールバック値の検証
            fallback_predictions = self.get_fallback_predictions()
            if not self.validate_predictions(fallback_predictions):
                errors.append("フォールバック予測値が無効です")

            # 範囲値の検証
            if not (0.0 <= self.MIN_PROBABILITY <= self.MAX_PROBABILITY <= 1.0):
                errors.append("確率範囲設定が無効です")

            if not (0.0 < self.PROBABILITY_SUM_MIN <= self.PROBABILITY_SUM_MAX <= 2.0):
                errors.append("確率合計範囲設定が無効です")

            # データ長の検証
            if not (1 <= self.DEFAULT_INDICATOR_LENGTH <= 100000):
                errors.append("デフォルト指標長が無効です")

        except Exception as e:
            errors.append(f"設定検証中にエラーが発生しました: {e}")

        return errors


@dataclass
class RetrainingConfig:
    """再学習関連の設定"""

    # スケジュール設定
    CHECK_INTERVAL_SECONDS: int = 3600
    MAX_CONCURRENT_JOBS: int = 2
    JOB_TIMEOUT_SECONDS: int = 7200

    # データ管理
    DATA_RETENTION_DAYS: int = 90
    INCREMENTAL_TRAINING_ENABLED: bool = True

    # 性能監視
    PERFORMANCE_DEGRADATION_THRESHOLD: float = 0.05
    DATA_DRIFT_THRESHOLD: float = 0.1


class MLConfig:
    """
    ML関連の統一設定クラス

    全てのML関連サービスがこのクラスを通じて設定にアクセスします。
    """

    def __init__(self):
        """設定の初期化"""
        self.data_processing = DataProcessingConfig()
        self.model = ModelConfig()
        self.lightgbm = LightGBMConfig()
        self.feature_engineering = FeatureEngineeringConfig()
        self.training = TrainingConfig()
        self.prediction = PredictionConfig()
        self.retraining = RetrainingConfig()

        # 設定の妥当性を検証
        self._validate_all_configs()

    def _validate_all_configs(self):
        """全設定の妥当性を検証"""
        try:
            # 予測設定の検証
            if not self.prediction.validate_config():
                errors = self.prediction.get_validation_errors()
                logging.warning(f"予測設定に問題があります: {errors}")

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

    def get_environment_info(self) -> Dict[str, Any]:
        """現在の環境設定情報を取得"""
        return {
            "debug_mode": self.data_processing.DEBUG_MODE,
            "log_level": self.data_processing.LOG_LEVEL,
            "max_ohlcv_rows": self.data_processing.MAX_OHLCV_ROWS,
            "feature_timeout": self.data_processing.FEATURE_CALCULATION_TIMEOUT,
            "default_predictions": self.prediction.get_default_predictions(),
            "validation_enabled": self.prediction.validate_config(),
        }

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
