"""
ML関連の例外クラス

MLサービスで使用される各種例外クラスを定義します。
"""


class MLBaseError(Exception):
    """ML関連エラーの基底クラス"""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class MLDataError(MLBaseError):
    """MLデータ関連のエラー
    
    データの形式、内容、品質に関する問題で発生する例外
    """
    
    def __init__(self, message: str, data_info: dict = None):
        self.data_info = data_info or {}
        super().__init__(message, "ML_DATA_ERROR")


class MLValidationError(MLBaseError):
    """MLバリデーション関連のエラー
    
    データやパラメータのバリデーション失敗時に発生する例外
    """
    
    def __init__(self, message: str, validation_details: dict = None):
        self.validation_details = validation_details or {}
        super().__init__(message, "ML_VALIDATION_ERROR")


class MLModelError(MLBaseError):
    """MLモデル関連のエラー
    
    モデルの読み込み、保存、予測実行時の問題で発生する例外
    """
    
    def __init__(self, message: str, model_info: dict = None):
        self.model_info = model_info or {}
        super().__init__(message, "ML_MODEL_ERROR")


class MLTrainingError(MLBaseError):
    """MLトレーニング関連のエラー
    
    モデルの訓練プロセスで発生する例外
    """
    
    def __init__(self, message: str, training_info: dict = None):
        self.training_info = training_info or {}
        super().__init__(message, "ML_TRAINING_ERROR")


class MLPredictionError(MLBaseError):
    """ML予測関連のエラー
    
    モデルの予測実行時に発生する例外
    """
    
    def __init__(self, message: str, prediction_info: dict = None):
        self.prediction_info = prediction_info or {}
        super().__init__(message, "ML_PREDICTION_ERROR")


class MLConfigurationError(MLBaseError):
    """ML設定関連のエラー
    
    ML設定の不正や不整合で発生する例外
    """
    
    def __init__(self, message: str, config_info: dict = None):
        self.config_info = config_info or {}
        super().__init__(message, "ML_CONFIGURATION_ERROR")


class MLResourceError(MLBaseError):
    """MLリソース関連のエラー
    
    メモリ不足、ディスク容量不足等のリソース問題で発生する例外
    """
    
    def __init__(self, message: str, resource_info: dict = None):
        self.resource_info = resource_info or {}
        super().__init__(message, "ML_RESOURCE_ERROR")


class MLTimeoutError(MLBaseError):
    """MLタイムアウト関連のエラー
    
    処理時間の制限を超過した場合に発生する例外
    """
    
    def __init__(self, message: str, timeout_info: dict = None):
        self.timeout_info = timeout_info or {}
        super().__init__(message, "ML_TIMEOUT_ERROR")


class MLFeatureError(MLBaseError):
    """ML特徴量関連のエラー
    
    特徴量の生成、選択、変換で発生する例外
    """
    
    def __init__(self, message: str, feature_info: dict = None):
        self.feature_info = feature_info or {}
        super().__init__(message, "ML_FEATURE_ERROR")
