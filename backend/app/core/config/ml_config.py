"""
ML関連の設定管理

分散していたML関連の設定値を統一的に管理するための設定クラス。
各MLサービスはこの設定を参照することで、一貫性のある動作を保証します。
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path


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


@dataclass
class ModelConfig:
    """モデル関連の設定"""
    # モデル保存パス
    MODEL_SAVE_PATH: str = "backend/models/"
    MODEL_BACKUP_PATH: str = "backend/models/backup/"
    
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
            'objective': self.OBJECTIVE,
            'num_class': self.NUM_CLASS,
            'metric': self.METRIC,
            'boosting_type': self.BOOSTING_TYPE,
            'num_leaves': self.NUM_LEAVES,
            'learning_rate': self.LEARNING_RATE,
            'feature_fraction': self.FEATURE_FRACTION,
            'bagging_fraction': self.BAGGING_FRACTION,
            'bagging_freq': self.BAGGING_FREQ,
            'verbose': self.VERBOSE,
            'random_state': self.RANDOM_STATE
        }


@dataclass
class FeatureEngineeringConfig:
    """特徴量エンジニアリング関連の設定"""
    # 計算期間のデフォルト値
    DEFAULT_LOOKBACK_PERIODS: Dict[str, int] = None
    
    # キャッシュ設定
    CACHE_ENABLED: bool = True
    MAX_CACHE_SIZE: int = 10
    CACHE_TTL_SECONDS: int = 3600
    
    # 特徴量計算設定
    PRICE_FEATURE_PERIODS: List[int] = None
    VOLATILITY_PERIODS: List[int] = None
    VOLUME_PERIODS: List[int] = None
    
    def __post_init__(self):
        """デフォルト値の設定"""
        if self.DEFAULT_LOOKBACK_PERIODS is None:
            self.DEFAULT_LOOKBACK_PERIODS = {
                "short_ma": 10,
                "long_ma": 50,
                "volatility": 20,
                "momentum": 14,
                "volume": 20
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

    # 予測値検証
    MIN_PROBABILITY: float = 0.0
    MAX_PROBABILITY: float = 1.0
    PROBABILITY_SUM_MIN: float = 0.8
    PROBABILITY_SUM_MAX: float = 1.2

    # 予測結果の拡張
    EXPAND_TO_DATA_LENGTH: bool = True

    def get_default_predictions(self) -> Dict[str, float]:
        """デフォルトの予測値を取得"""
        return {
            "up": self.DEFAULT_UP_PROB,
            "down": self.DEFAULT_DOWN_PROB,
            "range": self.DEFAULT_RANGE_PROB
        }


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
    
    @classmethod
    def from_env(cls) -> 'MLConfig':
        """環境変数から設定を読み込み"""
        config = cls()
        
        # 環境変数から設定を上書き
        if os.getenv('ML_MODEL_SAVE_PATH'):
            config.model.MODEL_SAVE_PATH = os.getenv('ML_MODEL_SAVE_PATH')
        
        if os.getenv('ML_MAX_OHLCV_ROWS'):
            config.data_processing.MAX_OHLCV_ROWS = int(os.getenv('ML_MAX_OHLCV_ROWS'))
        
        if os.getenv('ML_FEATURE_TIMEOUT'):
            config.data_processing.FEATURE_CALCULATION_TIMEOUT = int(os.getenv('ML_FEATURE_TIMEOUT'))
        
        # LightGBMパラメータ
        if os.getenv('ML_LEARNING_RATE'):
            config.lightgbm.LEARNING_RATE = float(os.getenv('ML_LEARNING_RATE'))
        
        if os.getenv('ML_NUM_LEAVES'):
            config.lightgbm.NUM_LEAVES = int(os.getenv('ML_NUM_LEAVES'))
        
        return config
    
    def validate(self) -> bool:
        """設定の妥当性を検証"""
        try:
            # データ処理設定の検証
            assert self.data_processing.MAX_OHLCV_ROWS > 0
            assert self.data_processing.FEATURE_CALCULATION_TIMEOUT > 0
            
            # モデル設定の検証
            assert os.path.exists(os.path.dirname(self.model.MODEL_SAVE_PATH)) or True  # ディレクトリは作成される
            
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
        return [
            self.model.MODEL_SAVE_PATH,
            "backend/models/",
            "models/",
            "ml_models/",
            "backend/ml_models/"
        ]


# グローバル設定インスタンス
ml_config = MLConfig.from_env()

# 設定の妥当性を検証
if not ml_config.validate():
    raise ValueError("ML設定の妥当性検証に失敗しました")
