"""
ML予測インターフェース

ML予測機能の標準インターフェースを定義します。
"""

from typing import Protocol, Dict, Any, Optional
import pandas as pd
import numpy as np


class MLPredictionInterface(Protocol):
    """
    ML予測機能の標準インターフェース
    
    このプロトコルを実装するクラスは、統一されたML予測APIを提供する必要があります。
    """

    def calculate_ml_indicators(
        self,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """
        ML予測確率指標を計算
        
        Args:
            df: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
        
        Returns:
            ML指標の辞書 {"ML_UP_PROB": array, "ML_DOWN_PROB": array, "ML_RANGE_PROB": array}
        
        Raises:
            MLDataError: 入力データが無効な場合
            MLValidationError: 結果の検証に失敗した場合
            MLModelError: モデル関連のエラーが発生した場合
        """
        ...

    def calculate_single_ml_indicator(
        self, indicator_type: str, df: pd.DataFrame
    ) -> np.ndarray:
        """
        単一のML指標を計算
        
        Args:
            indicator_type: 指標タイプ（ML_UP_PROB, ML_DOWN_PROB, ML_RANGE_PROB）
            df: OHLCVデータ
        
        Returns:
            指標値の配列
        
        Raises:
            MLDataError: 入力データが無効な場合
            ValueError: 未知の指標タイプが指定された場合
        """
        ...

    def predict_probabilities(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        特徴量から予測確率を計算
        
        Args:
            features: 特徴量データ
        
        Returns:
            予測確率の辞書 {"up": float, "down": float, "range": float}
        
        Raises:
            MLModelError: モデルが学習されていない場合
            MLDataError: 特徴量データが無効な場合
        """
        ...

    def load_model(self, model_path: str) -> bool:
        """
        学習済みモデルを読み込み
        
        Args:
            model_path: モデルファイルパス
        
        Returns:
            読み込み成功フラグ
        
        Raises:
            MLModelError: モデル読み込みに失敗した場合
            FileNotFoundError: モデルファイルが見つからない場合
        """
        ...

    def get_model_status(self) -> Dict[str, Any]:
        """
        モデルの状態を取得
        
        Returns:
            モデル状態の辞書
            - is_model_loaded: bool - モデルが読み込まれているか
            - is_trained: bool - モデルが学習済みか
            - last_predictions: Dict[str, float] - 最後の予測結果
            - feature_count: int - 特徴量数
            - model_type: str - モデルタイプ
        """
        ...

    def update_predictions(self, predictions: Dict[str, float]) -> None:
        """
        予測値を更新（外部から設定する場合）
        
        Args:
            predictions: 予測確率の辞書 {"up": float, "down": float, "range": float}
        
        Raises:
            MLValidationError: 予測値が無効な場合
        """
        ...

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得
        
        Args:
            top_n: 上位N個の特徴量
        
        Returns:
            特徴量重要度の辞書
        
        Raises:
            MLModelError: モデルが学習されていない場合
        """
        ...


class MLTrainingInterface(Protocol):
    """
    ML学習機能の標準インターフェース
    
    このプロトコルを実装するクラスは、統一されたML学習APIを提供する必要があります。
    """

    def train_model(
        self,
        training_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        **training_params: Any
    ) -> Dict[str, Any]:
        """
        MLモデルを学習
        
        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            **training_params: 学習パラメータ
        
        Returns:
            学習結果の辞書
            - success: bool - 学習成功フラグ
            - accuracy: float - 精度
            - precision: float - 適合率
            - recall: float - 再現率
            - f1_score: float - F1スコア
            - feature_count: int - 特徴量数
            - total_samples: int - 学習サンプル数
        
        Raises:
            MLDataError: 学習データが無効な場合
            MLModelError: 学習に失敗した場合
        """
        ...

    def save_model(self, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        学習済みモデルを保存
        
        Args:
            model_name: モデル名
            metadata: メタデータ（オプション）
        
        Returns:
            保存されたモデルのパス
        
        Raises:
            MLModelError: モデル保存に失敗した場合
        """
        ...


class MLServiceInterface(MLPredictionInterface, MLTrainingInterface, Protocol):
    """
    統合MLサービスインターフェース
    
    予測と学習の両方の機能を提供するサービスのインターフェース
    """
    pass


# 型エイリアス
MLIndicators = Dict[str, np.ndarray]
MLPredictions = Dict[str, float]
MLModelStatus = Dict[str, Any]
MLTrainingResult = Dict[str, Any]
