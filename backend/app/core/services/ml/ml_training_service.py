"""
ML学習サービス

MLモデルの学習機能を専門的に扱うサービス。
MLIndicatorServiceから学習機能を分離し、責任を明確化します。
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

from ...config.ml_config import ml_config
from ...utils.ml_error_handler import (
    MLErrorHandler, MLDataError, MLModelError,
    safe_ml_operation, ml_operation_context
)
from ..feature_engineering.feature_engineering_service import FeatureEngineeringService
from .signal_generator import MLSignalGenerator
from .model_manager import model_manager

logger = logging.getLogger(__name__)


class MLTrainingService:
    """
    ML学習サービス
    
    MLモデルの学習、評価、保存を専門的に行うサービス。
    """
    
    def __init__(self):
        """初期化"""
        self.config = ml_config
        self.feature_service = FeatureEngineeringService()
        self.ml_generator = MLSignalGenerator()
    
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
        with ml_operation_context("MLモデル学習"):
            try:
                # 入力データの検証
                self._validate_training_data(training_data)
                
                # 特徴量を計算
                features_df = self._calculate_features(
                    training_data, funding_rate_data, open_interest_data
                )
                
                # 学習用データを準備
                X, y = self._prepare_training_data(features_df, **training_params)
                
                # モデルを学習
                training_result = self._train_model(X, y, **training_params)
                
                # モデルを保存
                if save_model:
                    model_path = self._save_model(model_name or self.config.model.AUTO_STRATEGY_MODEL_NAME)
                    training_result["model_path"] = model_path
                
                # 学習結果を整形
                result = self._format_training_result(training_result, X, y)
                
                logger.info("MLモデル学習完了")
                return result
                
            except (MLDataError, MLModelError) as e:
                logger.error(f"MLモデル学習エラー: {e}")
                raise
            except Exception as e:
                logger.error(f"MLモデル学習で予期しないエラー: {e}")
                raise MLModelError(f"MLモデル学習に失敗しました: {e}")
    
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
        try:
            if not self.ml_generator.is_trained:
                raise MLModelError("評価対象の学習済みモデルがありません")
            
            # 特徴量を計算
            features_df = self._calculate_features(
                test_data, funding_rate_data, open_interest_data
            )
            
            # 予測を実行
            predictions = self.ml_generator.predict(features_df)
            
            # 評価結果を作成
            evaluation_result = {
                "predictions": predictions,
                "test_samples": len(test_data),
                "feature_count": len(self.ml_generator.feature_columns) if self.ml_generator.feature_columns else 0,
                "model_status": "trained" if self.ml_generator.is_trained else "not_trained"
            }
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"モデル評価エラー: {e}")
            raise MLModelError(f"モデル評価に失敗しました: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        学習状態を取得
        
        Returns:
            学習状態の辞書
        """
        return {
            "is_trained": self.ml_generator.is_trained,
            "feature_columns": self.ml_generator.feature_columns,
            "feature_count": len(self.ml_generator.feature_columns) if self.ml_generator.feature_columns else 0,
            "model_type": type(self.ml_generator.model).__name__ if self.ml_generator.model else None
        }
    
    def _validate_training_data(self, training_data: pd.DataFrame):
        """学習データの検証"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        MLErrorHandler.validate_dataframe(
            training_data,
            required_columns=required_columns,
            min_rows=self.config.training.MIN_TRAINING_SAMPLES
        )
    
    def _calculate_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """特徴量を計算"""
        return self.feature_service.calculate_advanced_features(
            ohlcv_data, funding_rate_data, open_interest_data
        )
    
    def _prepare_training_data(
        self, 
        features_df: pd.DataFrame, 
        **training_params
    ) -> tuple:
        """学習用データを準備"""
        # training_paramsから必要なパラメータを抽出
        prediction_horizon = training_params.get('prediction_horizon')
        threshold_up = training_params.get('threshold_up')
        threshold_down = training_params.get('threshold_down')
        
        return self.ml_generator.prepare_training_data(
            features_df,
            prediction_horizon=prediction_horizon,
            threshold_up=threshold_up,
            threshold_down=threshold_down
        )
    
    def _train_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        **training_params
    ) -> Dict[str, Any]:
        """モデルを学習"""
        # training_paramsから必要なパラメータを抽出
        test_size = training_params.get('test_size')
        random_state = training_params.get('random_state')
        
        return self.ml_generator.train(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
    
    def _save_model(self, model_name: str) -> str:
        """モデルを保存"""
        return self.ml_generator.save_model(model_name)
    
    def _format_training_result(
        self, 
        training_result: Dict[str, Any], 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Any]:
        """学習結果を整形"""
        return {
            "success": True,
            "accuracy": training_result.get("accuracy", 0.0),
            "loss": 1 - training_result.get("accuracy", 0.0),
            "feature_count": len(X.columns),
            "training_samples": training_result.get("train_samples", 0),
            "test_samples": training_result.get("test_samples", 0),
            "classification_report": training_result.get("classification_report", {}),
            "feature_importance": training_result.get("feature_importance", {}),
            "best_iteration": training_result.get("best_iteration")
        }
    
    @safe_ml_operation(default_value=False, error_message="モデル読み込みでエラーが発生しました")
    def load_model(self, model_path: str) -> bool:
        """
        学習済みモデルを読み込み
        
        Args:
            model_path: モデルファイルパス
        
        Returns:
            読み込み成功フラグ
        """
        return self.ml_generator.load_model(model_path)
    
    def get_latest_model_path(self) -> Optional[str]:
        """最新のモデルパスを取得"""
        return model_manager.get_latest_model("*")
    
    def list_available_models(self) -> list:
        """利用可能なモデルの一覧を取得"""
        return model_manager.list_models("*")


# グローバルインスタンス
ml_training_service = MLTrainingService()
