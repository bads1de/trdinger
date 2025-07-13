"""
ML学習サービス

MLモデルの学習機能を専門的に扱うサービス。
BaseMLTrainerを使用してコードの重複を解消し、責任を明確化します。
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

from .config import ml_config
from ...utils.ml_error_handler import (
    MLErrorHandler, MLDataError, MLModelError,
    safe_ml_operation, ml_operation_context
)
from .lightgbm_trainer import LightGBMTrainer
from .model_manager import model_manager

logger = logging.getLogger(__name__)


class MLTrainingService:
    """
    ML学習サービス

    BaseMLTrainerを使用してMLモデルの学習、評価、保存を専門的に行うサービス。
    コードの重複を解消し、保守性を向上させます。
    """

    def __init__(self, trainer_type: str = "lightgbm"):
        """
        初期化

        Args:
            trainer_type: 使用するトレーナーのタイプ（"lightgbm" または "randomforest"）
        """
        self.config = ml_config

        # トレーナーを選択
        if trainer_type.lower() == "lightgbm":
            self.trainer = LightGBMTrainer()
        elif trainer_type.lower() == "randomforest":
            from .lightgbm_trainer import RandomForestTrainer
            self.trainer = RandomForestTrainer()
        else:
            raise ValueError(f"サポートされていないトレーナータイプ: {trainer_type}")

        self.trainer_type = trainer_type
    
    def train_model(
        self,
        training_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        save_model: bool = True,
        model_name: Optional[str] = None,
        **training_params
    ) -> Dict[str, Any]:
        """
        MLモデルを学習

        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            save_model: モデルを保存するか
            model_name: モデル名（オプション）
            **training_params: 追加の学習パラメータ

        Returns:
            学習結果の辞書

        Raises:
            MLDataError: データが無効な場合
            MLModelError: 学習に失敗した場合
        """
        # BaseMLTrainerに委譲
        return self.trainer.train_model(
            training_data=training_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
            save_model=save_model,
            model_name=model_name,
            **training_params
        )
    
    def evaluate_model(
        self,
        test_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        学習済みモデルを評価

        Args:
            test_data: テスト用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）

        Returns:
            評価結果の辞書
        """
        # BaseMLTrainerに委譲
        return self.trainer.evaluate_model(
            test_data=test_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data
        )
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        学習状態を取得

        Returns:
            学習状態の辞書
        """
        if hasattr(self.trainer, 'get_model_info'):
            model_info = self.trainer.get_model_info()
            model_info["trainer_type"] = self.trainer_type
            return model_info
        else:
            return {
                "is_trained": self.trainer.is_trained,
                "feature_columns": self.trainer.feature_columns,
                "feature_count": len(self.trainer.feature_columns) if self.trainer.feature_columns else 0,
                "model_type": type(self.trainer.model).__name__ if self.trainer.model else None,
                "trainer_type": self.trainer_type
            }

    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        予測を実行

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測結果
        """
        predictions = self.trainer.predict(features_df)
        return {
            "predictions": predictions,
            "model_type": self.trainer_type,
            "feature_count": len(self.trainer.feature_columns) if self.trainer.feature_columns else 0
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Returns:
            特徴量重要度の辞書
        """
        if hasattr(self.trainer, 'get_feature_importance'):
            return self.trainer.get_feature_importance()
        else:
            return {}
    
    @safe_ml_operation(default_value=False, error_message="モデル読み込みでエラーが発生しました")
    def load_model(self, model_path: str) -> bool:
        """
        学習済みモデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        return self.trainer.load_model(model_path)

    def get_latest_model_path(self) -> Optional[str]:
        """最新のモデルパスを取得"""
        return model_manager.get_latest_model("*")

    def list_available_models(self) -> list:
        """利用可能なモデルの一覧を取得"""
        return model_manager.list_models("*")


# グローバルインスタンス（デフォルトはLightGBM）
ml_training_service = MLTrainingService(trainer_type="lightgbm")
