"""
ML関連の例外クラス

MLサービスで使用される各種例外クラスを定義します。
"""

from typing import Optional


class MLBaseError(Exception):
    """ML関連エラーの基底クラス"""
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message, self.error_code = message, error_code
        super().__init__(self.message)


class MLDataError(MLBaseError):
    """MLデータ関連のエラー"""
    def __init__(self, message: str, data_info: Optional[dict] = None):
        self.data_info = data_info or {}
        super().__init__(message, "ML_DATA_ERROR")


class MLValidationError(MLBaseError):
    """MLバリデーション関連のエラー"""
    def __init__(self, message: str, validation_details: Optional[dict] = None):
        self.validation_details = validation_details or {}
        super().__init__(message, "ML_VALIDATION_ERROR")


class MLModelError(MLBaseError):
    """MLモデル関連のエラー"""
    def __init__(self, message: str, model_info: Optional[dict] = None, error_code: str = "ML_MODEL_ERROR"):
        self.model_info = model_info or {}
        super().__init__(message, error_code)


class MLTrainingError(MLBaseError):
    """MLトレーニング関連のエラー"""
    def __init__(self, message: str, training_info: Optional[dict] = None):
        self.training_info = training_info or {}
        super().__init__(message, "ML_TRAINING_ERROR")


class MLPredictionError(MLBaseError):
    """ML予測関連のエラー"""
    def __init__(self, message: str, prediction_info: Optional[dict] = None):
        self.prediction_info = prediction_info or {}
        super().__init__(message, "ML_PREDICTION_ERROR")


class MLFeatureError(MLBaseError):
    """ML特徴量関連のエラー"""
    def __init__(self, message: str, feature_info: Optional[dict] = None):
        self.feature_info = feature_info or {}
        super().__init__(message, "ML_FEATURE_ERROR")



