"""
ML指標サービス

機械学習による予測確率を指標として提供するサービス。
BaseMLTrainerを使用してコードの重複を解消し、責任を明確化します。
"""

import logging
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional

from ...ml.lightgbm_trainer import LightGBMTrainer
from ...ml.model_manager import model_manager

logger = logging.getLogger(__name__)


class MLIndicatorService:
    """
    ML指標サービス

    BaseMLTrainerを使用して機械学習による予測確率を指標として提供します。
    コードの重複を解消し、保守性を向上させます。
    """

    def __init__(self, trainer: Optional[LightGBMTrainer] = None):
        """
        初期化

        Args:
            trainer: 使用するMLトレーナー（オプション）
        """
        self.trainer = trainer if trainer else LightGBMTrainer()
        self.is_model_loaded = self.trainer.is_trained
        self._last_predictions = {"up": 0.33, "down": 0.33, "range": 0.34}

        # 既存の学習済みモデルを自動読み込み
        if not self.is_model_loaded:
            self._try_load_latest_model()

    def calculate_ml_indicators(
        self,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
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
            logger.warning(
                f"大量のデータ（{len(df)}行）が提供されました。処理を制限します。"
            )
            df = df.tail(10000)  # 最新10,000行に制限

        try:
            # 必要なカラムの存在確認（大文字・小文字両方に対応）
            required_columns_lower = ["open", "high", "low", "close", "volume"]
            required_columns_upper = ["Open", "High", "Low", "Close", "Volume"]

            # 小文字のカラムが存在するかチェック
            missing_lower = [
                col for col in required_columns_lower if col not in df.columns
            ]
            # 大文字のカラムが存在するかチェック
            missing_upper = [
                col for col in required_columns_upper if col not in df.columns
            ]

            # どちらかのセットが完全に存在すればOK
            if len(missing_lower) == 0:
                # 小文字のカラムが揃っている
                df_normalized = df.copy()
            elif len(missing_upper) == 0:
                # 大文字のカラムが揃っている場合、小文字に変換
                df_normalized = df.copy()
                df_normalized.columns = [
                    col.lower() if col in required_columns_upper else col
                    for col in df_normalized.columns
                ]
            else:
                logger.error(
                    f"必要なカラムが不足: {missing_lower} (小文字) または {missing_upper} (大文字)"
                )
                return self._get_default_indicators(len(df))

            # 正規化されたデータフレームを使用
            df = df_normalized

            # 特徴量計算（タイムアウト付き）
            try:
                import platform
                import concurrent.futures

                # Windows環境ではconcurrent.futuresを使用、Unix系ではsignalを使用
                if platform.system() == "Windows":
                    # Windows環境でのタイムアウト処理
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=1
                    ) as executor:
                        future = executor.submit(
                            self.feature_service.calculate_advanced_features,
                            df,
                            funding_rate_data,
                            open_interest_data,
                        )
                        try:
                            features_df = future.result(
                                timeout=30
                            )  # 30秒のタイムアウト
                        except concurrent.futures.TimeoutError:
                            raise TimeoutError("特徴量計算がタイムアウトしました")
                else:
                    # Unix系環境でのシグナル処理
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
            ml_indicators = self._expand_predictions_to_data_length(
                predictions, data_length
            )

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
            success = self.trainer.load_model(model_path)
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
        save_model: bool = True,
        train_test_split: float = 0.8,
        cross_validation_folds: int = 5,
        random_state: int = 42,
        early_stopping_rounds: int = 100,
        max_depth: int = 10,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """
        MLモデルを学習

        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            save_model: モデルを保存するか
            train_test_split: トレーニング/テスト分割比率
            cross_validation_folds: クロスバリデーション分割数
            random_state: ランダムシード
            early_stopping_rounds: 早期停止ラウンド数
            max_depth: 最大深度
            n_estimators: 推定器数
            learning_rate: 学習率

        Returns:
            学習結果
        """
        try:
            # BaseMLTrainerに委譲
            result = self.trainer.train_model(
                training_data=training_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                save_model=save_model,
                model_name="auto_strategy_ml_model",
                test_size=1 - train_test_split,
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
            )

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
            "is_trained": self.trainer.is_trained,
            "last_predictions": self._last_predictions,
            "feature_count": (
                len(self.trainer.feature_columns) if self.trainer.feature_columns else 0
            ),
            "model_type": (
                self.trainer.model_type
                if hasattr(self.trainer, "model_type")
                else "Unknown"
            ),
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
        if self.is_model_loaded and self.trainer.is_trained:
            if hasattr(self.trainer, "get_feature_importance"):
                importance = self.trainer.get_feature_importance()
                # 上位N個を取得
                sorted_importance = sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                )
                return dict(sorted_importance[:top_n])
            else:
                return {}
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
            # データフレームの基本検証
            if df is None or df.empty:
                logger.warning(f"空のデータフレームが提供されました: {indicator_type}")
                return np.full(100, 0.33)

            ml_indicators = self.calculate_ml_indicators(df)

            if indicator_type in ml_indicators:
                return ml_indicators[indicator_type]
            else:
                logger.warning(f"未知のML指標タイプ: {indicator_type}")
                return np.full(len(df), 0.33)

        except Exception as e:
            logger.error(f"単一ML指標計算エラー {indicator_type}: {e}")
            return np.full(len(df) if df is not None and not df.empty else 100, 0.33)

    def _get_default_indicators(self, data_length: int) -> Dict[str, np.ndarray]:
        """デフォルトのML指標を取得"""
        default_value = 0.33
        return {
            "ML_UP_PROB": np.full(data_length, default_value),
            "ML_DOWN_PROB": np.full(data_length, default_value),
            "ML_RANGE_PROB": np.full(data_length, default_value + 0.01),
        }

    def _safe_ml_prediction(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """安全なML予測実行"""
        try:
            # モデルが読み込まれている場合のみ予測を実行
            if self.is_model_loaded and self.trainer.is_trained:
                # 特徴量を使用して予測
                predictions_array = self.trainer.predict(features_df)

                # 予測結果を正規化（3クラス分類: [下落, レンジ, 上昇]）
                if predictions_array.ndim == 2 and predictions_array.shape[1] == 3:
                    # 既に3クラス分類の確率 - 最後の行の予測値を使用
                    last_prediction = predictions_array[-1]
                    predictions = {
                        "down": float(last_prediction[0]),
                        "range": float(last_prediction[1]),
                        "up": float(last_prediction[2]),
                    }
                elif predictions_array.ndim == 1:
                    # バイナリ分類の場合は3クラスに変換
                    last_prediction = predictions_array[-1]
                    predictions = {
                        "up": float(last_prediction),
                        "down": float(1 - last_prediction),
                        "range": 0.0,
                    }
                else:
                    # デフォルト値
                    predictions = self._last_predictions.copy()

                # 予測値の妥当性チェック
                if self._validate_predictions(predictions):
                    self._last_predictions = predictions
                    return predictions
                else:
                    logger.warning("予測値が無効、前回の予測値を使用")
                    return self._last_predictions
            else:
                logger.warning(
                    "MLモデルが読み込まれていません。デフォルト値を使用します。"
                )
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

            return {
                "ML_UP_PROB": ml_up_prob,
                "ML_DOWN_PROB": ml_down_prob,
                "ML_RANGE_PROB": ml_range_prob,
            }

        except Exception as e:
            logger.error(f"予測値拡張エラー: {e}")
            return self._get_default_indicators(data_length)

    def _preprocess_training_data(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        トレーニングデータの前処理

        Args:
            ohlcv_data: OHLCVデータ
            funding_rate_data: ファンディングレートデータ
            open_interest_data: オープンインタレストデータ

        Returns:
            前処理済みデータ
        """
        # 基本的なOHLCVデータをコピー
        processed = ohlcv_data.copy()

        # カラム名を小文字に統一
        processed.columns = [col.lower() for col in processed.columns]

        # 追加データがある場合は結合
        if funding_rate_data is not None and not funding_rate_data.empty:
            funding_rate_data.columns = [
                col.lower() for col in funding_rate_data.columns
            ]
            processed = processed.join(funding_rate_data, how="left")

        if open_interest_data is not None and not open_interest_data.empty:
            open_interest_data.columns = [
                col.lower() for col in open_interest_data.columns
            ]
            processed = processed.join(open_interest_data, how="left")

        # 欠損値を前方補完
        processed = processed.fillna(method="ffill").fillna(method="bfill")

        return processed

    def _create_features_and_targets(self, data: pd.DataFrame) -> tuple:
        """
        特徴量とターゲットを作成

        Args:
            data: 前処理済みデータ

        Returns:
            (特徴量DataFrame, ターゲットSeries)
        """
        # 基本的なテクニカル指標を計算
        features = pd.DataFrame(index=data.index)

        # 価格変動率
        features["price_change"] = data["close"].pct_change()
        features["high_low_ratio"] = (data["high"] - data["low"]) / data["close"]
        features["volume_change"] = data["volume"].pct_change()

        # 移動平均
        for period in [5, 10, 20]:
            features[f"sma_{period}"] = data["close"].rolling(period).mean()
            features[f"price_sma_{period}_ratio"] = (
                data["close"] / features[f"sma_{period}"]
            )

        # RSI
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features["rsi"] = 100 - (100 / (1 + rs))

        # ボリンジャーバンド
        sma_20 = data["close"].rolling(20).mean()
        std_20 = data["close"].rolling(20).std()
        features["bb_upper"] = sma_20 + (std_20 * 2)
        features["bb_lower"] = sma_20 - (std_20 * 2)
        features["bb_position"] = (data["close"] - features["bb_lower"]) / (
            features["bb_upper"] - features["bb_lower"]
        )

        # 追加データがある場合の特徴量
        if "fundingrate" in data.columns:
            features["funding_rate"] = data["fundingrate"]
            features["funding_rate_change"] = data["fundingrate"].pct_change()

        if "openinterest" in data.columns:
            features["open_interest"] = data["openinterest"]
            features["oi_change"] = data["openinterest"].pct_change()

        # ターゲット作成（24時間後の価格変動）
        future_returns = data["close"].shift(-24).pct_change(24)

        # 3クラス分類：上昇(0)、下降(1)、横ばい(2)
        targets = pd.Series(index=data.index, dtype=int)
        targets[future_returns > 0.02] = 0  # 上昇
        targets[future_returns < -0.02] = 1  # 下降
        targets[(future_returns >= -0.02) & (future_returns <= 0.02)] = 2  # 横ばい

        # 欠損値を除去
        valid_idx = features.dropna().index.intersection(targets.dropna().index)
        features = features.loc[valid_idx]
        targets = targets.loc[valid_idx]

        return features, targets

    def _save_model(self) -> str:
        """
        モデルを保存

        Returns:
            保存されたモデルのパス
        """
        import joblib
        import os
        from datetime import datetime

        # モデル保存ディレクトリを作成
        model_dir = "ml_models"
        os.makedirs(model_dir, exist_ok=True)

        # ファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"ml_model_{timestamp}.joblib")

        # モデルと特徴量カラムを保存
        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "timestamp": timestamp,
        }

        joblib.dump(model_data, model_path)
        logger.info(f"MLモデルを保存しました: {model_path}")

        return model_path

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

        except Exception as e:
            logger.error(f"ML指標の妥当性検証エラー: {e}")
            return False

    def _try_load_latest_model(self) -> bool:
        """
        最新の学習済みモデルを自動読み込み

        Returns:
            読み込み成功フラグ
        """
        try:
            # ModelManagerを使用して最新のモデルを取得
            latest_model_path = model_manager.get_latest_model("*")

            if not latest_model_path:
                logger.info(
                    "学習済みMLモデルが見つかりません。ML機能はデフォルト値で動作します。"
                )
                return False

            # モデルを読み込み
            success = self.trainer.load_model(latest_model_path)
            if success:
                self.is_model_loaded = True
                logger.info(
                    f"最新のMLモデルを自動読み込みしました: {os.path.basename(latest_model_path)}"
                )
                return True
            else:
                logger.warning(
                    f"MLモデルの読み込みに失敗しました: {os.path.basename(latest_model_path)}"
                )
                return False

        except Exception as e:
            logger.warning(f"MLモデルの自動読み込み中にエラーが発生しました: {e}")
            return False

        except Exception:
            return False
