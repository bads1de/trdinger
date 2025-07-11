"""
ML指標サービス

機械学習による予測確率を指標として提供するサービス。
FeatureEngineeringServiceとMLSignalGeneratorを統合して、
GAで使用可能なML予測指標を生成します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from ...feature_engineering.feature_engineering_service import FeatureEngineeringService
from ...ml.signal_generator import MLSignalGenerator

logger = logging.getLogger(__name__)


class MLIndicatorService:
    """
    ML指標サービス
    
    機械学習による予測確率を指標として提供します。
    """

    def __init__(self):
        """初期化"""
        self.feature_service = FeatureEngineeringService()
        self.ml_generator = MLSignalGenerator()
        self.is_model_loaded = False
        self._last_predictions = {"up": 0.33, "down": 0.33, "range": 0.34}

    def calculate_ml_indicators(
        self,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        ML予測確率指標を計算（強化されたエラーハンドリング付き）

        Args:
            df: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）

        Returns:
            ML指標の辞書 {"ML_UP_PROB": array, "ML_DOWN_PROB": array, "ML_RANGE_PROB": array}
        """
        # 入力データの検証
        if df is None or df.empty:
            logger.warning("空のOHLCVデータが提供されました")
            return self._get_default_indicators(100)

        # メモリ使用量チェック
        if len(df) > 10000:
            logger.warning(f"大量のデータ（{len(df)}行）が提供されました。処理を制限します。")
            df = df.tail(10000)  # 最新10,000行に制限

        try:
            # 必要なカラムの存在確認
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"必要なカラムが不足: {missing_columns}")
                return self._get_default_indicators(len(df))

            # 特徴量計算（タイムアウト付き）
            try:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("特徴量計算がタイムアウトしました")

                # 30秒のタイムアウトを設定
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)

                features_df = self.feature_service.calculate_advanced_features(
                    df, funding_rate_data, open_interest_data
                )

                signal.alarm(0)  # タイムアウトをクリア

            except TimeoutError:
                logger.error("特徴量計算がタイムアウトしました")
                return self._get_default_indicators(len(df))
            except Exception as e:
                logger.error(f"特徴量計算エラー: {e}")
                return self._get_default_indicators(len(df))

            # ML予測の実行
            predictions = self._safe_ml_prediction(features_df)

            # 予測確率を全データ長に拡張
            data_length = len(df)
            ml_indicators = self._expand_predictions_to_data_length(predictions, data_length)

            # 結果の妥当性チェック
            if not self._validate_ml_indicators(ml_indicators):
                logger.warning("ML指標の妥当性チェックに失敗、デフォルト値を使用")
                return self._get_default_indicators(data_length)

            return ml_indicators

        except MemoryError:
            logger.error("メモリ不足によりML指標計算に失敗")
            return self._get_default_indicators(len(df))
        except Exception as e:
            logger.error(f"予期しないML指標計算エラー: {e}")
            return self._get_default_indicators(len(df))

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

    def train_model(
        self,
        training_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        MLモデルを学習

        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            save_model: モデルを保存するか

        Returns:
            学習結果
        """
        try:
            # 特徴量を計算
            features_df = self.feature_service.calculate_advanced_features(
                training_data, funding_rate_data, open_interest_data
            )

            # 学習用データを準備
            X, y = self.ml_generator.prepare_training_data(features_df)

            # モデルを学習
            result = self.ml_generator.train(X, y)

            # モデルを保存
            if save_model:
                model_path = self.ml_generator.save_model("auto_strategy_ml_model")
                result["model_path"] = model_path

            self.is_model_loaded = True
            logger.info("MLモデル学習完了")

            return result

        except Exception as e:
            logger.error(f"MLモデル学習エラー: {e}")
            raise

    def get_model_status(self) -> Dict[str, Any]:
        """
        モデルの状態を取得

        Returns:
            モデル状態の辞書
        """
        return {
            "is_model_loaded": self.is_model_loaded,
            "is_trained": self.ml_generator.is_trained if hasattr(self.ml_generator, 'is_trained') else False,
            "last_predictions": self._last_predictions,
            "feature_count": len(self.ml_generator.feature_columns) if self.ml_generator.feature_columns else 0
        }

    def update_predictions(self, predictions: Dict[str, float]):
        """
        予測値を更新（外部から設定する場合）

        Args:
            predictions: 予測確率の辞書
        """
        if all(key in predictions for key in ["up", "down", "range"]):
            self._last_predictions = predictions
            logger.debug(f"ML予測値を更新: {predictions}")
        else:
            logger.warning("無効な予測値形式です")

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        if self.is_model_loaded and self.ml_generator.is_trained:
            return self.ml_generator.get_feature_importance(top_n)
        else:
            return {}

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
            ml_indicators = self.calculate_ml_indicators(df)
            
            if indicator_type in ml_indicators:
                return ml_indicators[indicator_type]
            else:
                logger.warning(f"未知のML指標タイプ: {indicator_type}")
                return np.full(len(df), 0.33)

        except Exception as e:
            logger.error(f"単一ML指標計算エラー {indicator_type}: {e}")
            return np.full(len(df), 0.33)

    def _get_default_indicators(self, data_length: int) -> Dict[str, np.ndarray]:
        """デフォルトのML指標を取得"""
        default_value = 0.33
        return {
            "ML_UP_PROB": np.full(data_length, default_value),
            "ML_DOWN_PROB": np.full(data_length, default_value),
            "ML_RANGE_PROB": np.full(data_length, default_value + 0.01)
        }

    def _safe_ml_prediction(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """安全なML予測実行"""
        try:
            # モデルが学習済みの場合は予測を実行
            if self.is_model_loaded and self.ml_generator.is_trained:
                predictions = self.ml_generator.predict(features_df)

                # 予測値の妥当性チェック
                if self._validate_predictions(predictions):
                    self._last_predictions = predictions
                    return predictions
                else:
                    logger.warning("予測値が無効、前回の予測値を使用")
                    return self._last_predictions
            else:
                # モデルが未学習の場合はデフォルト値を使用
                logger.debug("MLモデルが未学習のため、デフォルト予測値を使用")
                return self._last_predictions

        except Exception as e:
            logger.warning(f"ML予測エラー、前回の予測値を使用: {e}")
            return self._last_predictions

    def _validate_predictions(self, predictions: Dict[str, float]) -> bool:
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
                if not (0 <= value <= 1):
                    return False

            # 合計が妥当な範囲内か（0.8-1.2）
            total = sum(predictions[key] for key in required_keys)
            if not (0.8 <= total <= 1.2):
                return False

            return True

        except Exception:
            return False

    def _expand_predictions_to_data_length(
        self, predictions: Dict[str, float], data_length: int
    ) -> Dict[str, np.ndarray]:
        """予測値をデータ長に拡張"""
        try:
            ml_up_prob = np.full(data_length, predictions["up"])
            ml_down_prob = np.full(data_length, predictions["down"])
            ml_range_prob = np.full(data_length, predictions["range"])

            # 最新の数ポイントのみ実際の予測値を使用（計算負荷軽減）
            if data_length > 10:
                # 古いデータは中立値を使用
                neutral_value = 0.33
                ml_up_prob[:-10] = neutral_value
                ml_down_prob[:-10] = neutral_value
                ml_range_prob[:-10] = neutral_value + 0.01

            return {
                "ML_UP_PROB": ml_up_prob,
                "ML_DOWN_PROB": ml_down_prob,
                "ML_RANGE_PROB": ml_range_prob
            }

        except Exception as e:
            logger.error(f"予測値拡張エラー: {e}")
            return self._get_default_indicators(data_length)

    def _validate_ml_indicators(self, ml_indicators: Dict[str, np.ndarray]) -> bool:
        """ML指標の妥当性を検証"""
        try:
            required_indicators = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]

            # 必要な指標が存在するか
            if not all(indicator in ml_indicators for indicator in required_indicators):
                return False

            # 各指標の妥当性をチェック
            for indicator, values in ml_indicators.items():
                if not isinstance(values, np.ndarray):
                    return False
                if len(values) == 0:
                    return False
                if not np.all((values >= 0) & (values <= 1)):
                    return False
                if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                    return False

            return True

        except Exception:
            return False
