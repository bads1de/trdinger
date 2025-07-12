"""
ML オーケストレーター

MLIndicatorServiceの責任を分離し、軽量化したオーケストレーター。
各ML関連サービスを統合し、一貫性のあるML指標を提供します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from ....config.ml_config import ml_config
from ....utils.ml_error_handler import (
    MLErrorHandler, MLDataError, MLValidationError,
    timeout_decorator, safe_ml_operation, ml_operation_context
)
from ...ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
from ...ml.signal_generator import MLSignalGenerator
from ...ml.model_manager import model_manager

logger = logging.getLogger(__name__)


class MLOrchestrator:
    """
    ML オーケストレーター
    
    責任を分離した軽量なMLサービス統合クラス。
    特徴量計算、モデル予測、結果の統合を行います。
    """
    
    def __init__(self, ml_signal_generator: Optional[MLSignalGenerator] = None):
        """
        初期化
        
        Args:
            ml_signal_generator: MLSignalGeneratorインスタンス（オプション）
        """
        self.config = ml_config
        self.feature_service = FeatureEngineeringService()
        self.ml_generator = ml_signal_generator if ml_signal_generator else MLSignalGenerator()
        self.is_model_loaded = False
        self._last_predictions = self.config.prediction.get_default_predictions()
        
        # 既存の学習済みモデルを自動読み込み
        self._try_load_latest_model()
    
    @timeout_decorator(timeout_seconds=ml_config.data_processing.FEATURE_CALCULATION_TIMEOUT)
    def calculate_ml_indicators(
        self,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        ML予測確率指標を計算
        
        Args:
            df: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
        
        Returns:
            ML指標の辞書 {"ML_UP_PROB": array, "ML_DOWN_PROB": array, "ML_RANGE_PROB": array}
        """
        with ml_operation_context("ML指標計算"):
            try:
                # 入力データの検証
                self._validate_input_data(df)
                
                # データサイズ制限
                df = self._limit_data_size(df)
                
                # カラム名の正規化
                df = self._normalize_column_names(df)
                
                # 特徴量計算
                features_df = self._calculate_features(df, funding_rate_data, open_interest_data)
                
                # ML予測の実行
                predictions = self._safe_ml_prediction(features_df)
                
                # 予測確率を全データ長に拡張
                ml_indicators = self._expand_predictions_to_data_length(predictions, len(df))
                
                # 結果の妥当性チェック
                self._validate_ml_indicators(ml_indicators)
                
                return ml_indicators
                
            except (MLDataError, MLValidationError) as e:
                logger.warning(f"ML指標計算で検証エラー: {e}")
                return self._get_default_indicators(len(df))
            except Exception as e:
                logger.error(f"ML指標計算で予期しないエラー: {e}")
                return self._get_default_indicators(len(df))
    
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
        """
        try:
            # データフレームの基本検証
            if df is None or df.empty:
                logger.warning(f"空のデータフレームが提供されました: {indicator_type}")
                return np.full(100, self.config.prediction.DEFAULT_UP_PROB)
            
            ml_indicators = self.calculate_ml_indicators(df)
            
            if indicator_type in ml_indicators:
                return ml_indicators[indicator_type]
            else:
                logger.warning(f"未知のML指標タイプ: {indicator_type}")
                return np.full(len(df), self.config.prediction.DEFAULT_UP_PROB)
                
        except Exception as e:
            logger.error(f"単一ML指標計算エラー {indicator_type}: {e}")
            default_length = len(df) if df is not None and not df.empty else 100
            return np.full(default_length, self.config.prediction.DEFAULT_UP_PROB)
    
    def load_model(self, model_path: str) -> bool:
        """
        学習済みモデルを読み込み
        
        Args:
            model_path: モデルファイルパス
        
        Returns:
            読み込み成功フラグ
        """
        try:
            success = self.ml_generator.load_model(model_path)
            if success:
                self.is_model_loaded = True
                logger.info(f"MLモデル読み込み成功: {model_path}")
            else:
                logger.warning(f"MLモデル読み込み失敗: {model_path}")
            return success
            
        except Exception as e:
            logger.error(f"MLモデル読み込みエラー: {e}")
            return False
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        モデルの状態を取得
        
        Returns:
            モデル状態の辞書
        """
        return {
            "is_model_loaded": self.is_model_loaded,
            "is_trained": getattr(self.ml_generator, 'is_trained', False),
            "last_predictions": self._last_predictions,
            "feature_count": len(self.ml_generator.feature_columns) if self.ml_generator.feature_columns else 0
        }
    
    def update_predictions(self, predictions: Dict[str, float]):
        """
        予測値を更新（外部から設定する場合）
        
        Args:
            predictions: 予測確率の辞書
        """
        try:
            MLErrorHandler.validate_predictions(predictions)
            self._last_predictions = predictions
            logger.debug(f"ML予測値を更新: {predictions}")
        except MLValidationError as e:
            logger.warning(f"無効な予測値形式: {e}")
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得
        
        Args:
            top_n: 上位N個の特徴量
        
        Returns:
            特徴量重要度の辞書
        """
        if self.is_model_loaded and getattr(self.ml_generator, 'is_trained', False):
            return self.ml_generator.get_feature_importance(top_n)
        else:
            return {}
    
    def _validate_input_data(self, df: pd.DataFrame):
        """入力データの検証"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        MLErrorHandler.validate_dataframe(
            df, 
            required_columns=required_columns,
            min_rows=1
        )
    
    def _limit_data_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """データサイズを制限"""
        max_rows = self.config.data_processing.MAX_OHLCV_ROWS
        if len(df) > max_rows:
            logger.warning(f"大量のデータ（{len(df)}行）、最新{max_rows}行に制限")
            return df.tail(max_rows)
        return df
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """カラム名を正規化"""
        # 必要なカラムの存在確認（大文字・小文字両方に対応）
        required_columns_lower = ['open', 'high', 'low', 'close', 'volume']
        required_columns_upper = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 小文字のカラムが存在するかチェック
        missing_lower = [col for col in required_columns_lower if col not in df.columns]
        # 大文字のカラムが存在するかチェック
        missing_upper = [col for col in required_columns_upper if col not in df.columns]
        
        # どちらかのセットが完全に存在すればOK
        if len(missing_lower) == 0:
            # 小文字のカラムが揃っている
            return df.copy()
        elif len(missing_upper) == 0:
            # 大文字のカラムが揃っている場合、小文字に変換
            df_normalized = df.copy()
            df_normalized.columns = [
                col.lower() if col in required_columns_upper else col 
                for col in df_normalized.columns
            ]
            return df_normalized
        else:
            raise MLDataError(f"必要なカラムが不足: {missing_lower} (小文字) または {missing_upper} (大文字)")
    
    def _calculate_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """特徴量を計算"""
        return self.feature_service.calculate_advanced_features(
            df, funding_rate_data, open_interest_data
        )
    
    @safe_ml_operation(
        default_value=ml_config.prediction.get_default_predictions(),
        error_message="ML予測でエラーが発生しました"
    )
    def _safe_ml_prediction(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """安全なML予測実行"""
        # 予測を実行
        predictions = self.ml_generator.predict(features_df)
        
        # 予測値の妥当性チェック
        MLErrorHandler.validate_predictions(predictions)
        self._last_predictions = predictions
        return predictions
    
    def _expand_predictions_to_data_length(
        self, predictions: Dict[str, float], data_length: int
    ) -> Dict[str, np.ndarray]:
        """予測値をデータ長に拡張"""
        try:
            return {
                "ML_UP_PROB": np.full(data_length, predictions["up"]),
                "ML_DOWN_PROB": np.full(data_length, predictions["down"]),
                "ML_RANGE_PROB": np.full(data_length, predictions["range"])
            }
        except Exception as e:
            logger.error(f"予測値拡張エラー: {e}")
            return self._get_default_indicators(data_length)
    
    def _validate_ml_indicators(self, ml_indicators: Dict[str, np.ndarray]):
        """ML指標の妥当性を検証"""
        required_indicators = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
        
        # 必要な指標が存在するか
        missing_indicators = [ind for ind in required_indicators if ind not in ml_indicators]
        if missing_indicators:
            raise MLValidationError(f"必要なML指標が不足: {missing_indicators}")
        
        # 各指標の妥当性をチェック
        for indicator, values in ml_indicators.items():
            if not isinstance(values, np.ndarray):
                raise MLValidationError(f"ML指標が配列ではありません: {indicator}")
            if len(values) == 0:
                raise MLValidationError(f"ML指標が空です: {indicator}")
            if not np.all((values >= 0) & (values <= 1)):
                raise MLValidationError(f"ML指標の値が範囲外です: {indicator}")
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                raise MLValidationError(f"ML指標に無効な値が含まれています: {indicator}")
    
    def _get_default_indicators(self, data_length: int) -> Dict[str, np.ndarray]:
        """デフォルトのML指標を取得"""
        config = self.config.prediction
        return {
            "ML_UP_PROB": np.full(data_length, config.DEFAULT_UP_PROB),
            "ML_DOWN_PROB": np.full(data_length, config.DEFAULT_DOWN_PROB),
            "ML_RANGE_PROB": np.full(data_length, config.DEFAULT_RANGE_PROB)
        }
    
    def _try_load_latest_model(self) -> bool:
        """最新の学習済みモデルを自動読み込み"""
        try:
            latest_model = model_manager.get_latest_model("*")
            if latest_model:
                success = self.ml_generator.load_model(latest_model)
                if success:
                    self.is_model_loaded = True
                    logger.info(f"最新のMLモデルを自動読み込み: {latest_model}")
                    return True
                else:
                    logger.warning(f"MLモデルの読み込みに失敗: {latest_model}")
            else:
                logger.info("学習済みMLモデルが見つかりません。ML機能はデフォルト値で動作します。")
            return False
            
        except Exception as e:
            logger.warning(f"MLモデルの自動読み込み中にエラー: {e}")
            return False
