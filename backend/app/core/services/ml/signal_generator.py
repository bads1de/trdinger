"""
ML信号生成器

LightGBMを用いた3クラス分類モデル（上昇・下落・レンジ）による
価格予測信号を生成します。
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import lightgbm as lgb

from .config import ml_config
from ...utils.ml_error_handler import MLModelError
from .model_manager import model_manager

logger = logging.getLogger(__name__)


class MLSignalGenerator:
    """
    ML信号生成器

    特徴量エンジニアリングサービスが生成した特徴量を使用して、
    未来の価格動向（上昇・下落・レンジ）を予測するモデルを構築・運用します。
    """

    def __init__(self, model_save_path: Optional[str] = None):
        """
        初期化

        Args:
            model_save_path: モデル保存パス（オプション、設定から取得）
        """
        self.config = ml_config
        self.model_save_path = model_save_path or self.config.model.MODEL_SAVE_PATH
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False

        # モデル保存ディレクトリを作成
        os.makedirs(self.model_save_path, exist_ok=True)

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        prediction_horizon: Optional[int] = None,
        threshold_up: Optional[float] = None,
        threshold_down: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        学習用データを準備

        Args:
            df: 特徴量付きDataFrame
            prediction_horizon: 予測期間（時間）
            threshold_up: 上昇判定閾値
            threshold_down: 下落判定閾値

        Returns:
            特徴量DataFrame、ラベルSeries
        """
        try:
            # 設定からデフォルト値を取得
            if prediction_horizon is None:
                prediction_horizon = self.config.training.PREDICTION_HORIZON
            if threshold_up is None:
                threshold_up = self.config.training.THRESHOLD_UP
            if threshold_down is None:
                threshold_down = self.config.training.THRESHOLD_DOWN
            # ターゲットカラムの存在チェック
            required_target_columns = ["target_up", "target_down", "target_range"]
            if not all(col in df.columns for col in required_target_columns):
                raise ValueError("Target columns not found in features_df")

            # 未来の価格変化率を計算
            future_returns = df["close"].shift(-prediction_horizon) / df["close"] - 1

            # 3クラス分類のラベルを作成
            # float型でラベルを初期化し、NaNを許容
            labels = pd.Series(np.nan, index=df.index, dtype=float)
            labels[future_returns >= threshold_up] = 2  # 上昇
            labels[future_returns <= threshold_down] = 0  # 下落
            labels[
                (future_returns > threshold_down) & (future_returns < threshold_up)
            ] = 1  # レンジ

            # NaNを除去
            valid_mask = labels.notna()
            df_clean = df[valid_mask].copy()
            # NaN除去後にint型に変換
            labels_clean = labels[valid_mask].astype(int)

            # 特徴量カラムを選択（数値カラムのみ）
            feature_columns = []
            for col in df_clean.columns:
                if (
                    col not in ["timestamp", "open", "high", "low", "close"]
                    and col not in required_target_columns
                ):
                    if df_clean[col].dtype in ["int64", "float64"]:
                        feature_columns.append(col)

            self.feature_columns = feature_columns
            features_df = df_clean[feature_columns].fillna(0)

            logger.info(
                f"学習データ準備完了: {len(features_df)}サンプル, {len(feature_columns)}特徴量"
            )
            logger.info(
                f"ラベル分布: 下落={sum(labels_clean==0)}, レンジ={sum(labels_clean==1)}, 上昇={sum(labels_clean==2)}"
            )

            return features_df, labels_clean

        except Exception as e:
            logger.error(f"学習データ準備エラー: {e}")
            raise

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        モデルを学習

        Args:
            features: 特徴量DataFrame
            labels: ラベルSeries
            test_size: テストデータの割合
            random_state: 乱数シード

        Returns:
            学習結果の辞書
        """
        try:
            # 設定からデフォルト値を取得
            if test_size is None:
                test_size = 1 - self.config.training.TRAIN_TEST_SPLIT
            if random_state is None:
                random_state = self.config.training.RANDOM_STATE
            # 時系列分割（リークを防ぐため）
            split_index = int(len(features) * (1 - test_size))
            X_train = features.iloc[:split_index]
            X_test = features.iloc[split_index:]
            y_train = labels.iloc[:split_index]
            y_test = labels.iloc[split_index:]

            # 特徴量の標準化
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # LightGBMデータセットを作成
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

            # LightGBMパラメータ（設定から取得）
            params = self.config.lightgbm.to_dict()
            params["random_state"] = random_state

            # モデル学習
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                num_boost_round=self.config.lightgbm.NUM_BOOST_ROUND,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.config.lightgbm.EARLY_STOPPING_ROUNDS
                    ),
                    lgb.log_evaluation(0),
                ],
            )

            # 予測と評価
            y_pred_proba = self.model.predict(
                X_test_scaled, num_iteration=self.model.best_iteration
            )
            # 型チェッカーのためにキャスト
            y_pred_proba = np.array(y_pred_proba)
            y_pred_class = np.argmax(y_pred_proba, axis=1)

            # 基本的な評価指標を計算
            accuracy = accuracy_score(y_test, y_pred_class)
            class_report = classification_report(
                y_test, y_pred_class, output_dict=True, zero_division=0.0
            )

            # 詳細な性能指標を計算
            precision = precision_score(
                y_test, y_pred_class, average="weighted", zero_division=0.0
            )
            recall = recall_score(
                y_test, y_pred_class, average="weighted", zero_division=0.0
            )
            f1 = f1_score(y_test, y_pred_class, average="weighted", zero_division=0.0)

            # AUCスコアを計算（多クラス分類の場合はOvR方式）
            try:
                if len(np.unique(y_test)) > 2:
                    # 多クラス分類の場合
                    auc = roc_auc_score(
                        y_test, y_pred_proba, multi_class="ovr", average="weighted"
                    )
                else:
                    # 二値分類の場合
                    auc = roc_auc_score(
                        y_test,
                        y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba,
                    )
            except Exception as e:
                logger.warning(f"AUCスコア計算エラー: {e}")
                auc = 0.0

            # 特徴量重要度
            feature_importance = {}
            if self.feature_columns and hasattr(self.model, "feature_importance"):
                importances = self.model.feature_importance(importance_type="gain")
                feature_importance = dict(zip(self.feature_columns, importances))

            self.is_trained = True

            result = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc_score": auc,
                "classification_report": class_report,
                "feature_importance": feature_importance,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "best_iteration": self.model.best_iteration,
            }

            logger.info(f"モデル学習完了: 精度={accuracy:.4f}")
            return result

        except Exception as e:
            logger.error(f"モデル学習エラー: {e}")
            raise

    def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        予測を実行

        Args:
            features: 特徴量DataFrame

        Returns:
            予測確率の辞書 {"up": float, "down": float, "range": float}
        """
        try:
            if not self.is_trained or self.model is None:
                # モデル未学習時は警告レベルでログ出力（エラーレベルから変更）
                logger.warning("モデルが学習されていません。デフォルト値を返します。")
                return self.config.prediction.get_default_predictions()  # 早期リターン

            if self.feature_columns is None:
                # 特徴量カラムが設定されていない場合、利用可能な全カラムを使用
                logger.warning(
                    "特徴量カラムが設定されていません。利用可能な全カラムを使用します。"
                )
                features_selected = features.fillna(0)
            else:
                # 特徴量を選択・整形
                available_columns = [
                    col for col in self.feature_columns if col in features.columns
                ]
                if not available_columns:
                    logger.warning(
                        "指定された特徴量カラムが見つかりません。利用可能な全カラムを使用します。"
                    )
                    features_selected = features.fillna(0)
                else:
                    features_selected = features[available_columns].fillna(0)

            # 標準化
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_selected)
            else:
                logger.warning(
                    "スケーラーが設定されていません。標準化をスキップします。"
                )
                features_scaled = features_selected.values

            # 予測（モデルタイプに応じて適切なメソッドを使用）
            # LightGBMモデルの場合
            predictions = np.array(
                self.model.predict(
                    features_scaled, num_iteration=self.model.best_iteration
                )
            )

            # 最新の予測結果を取得
            if predictions.ndim == 2:
                latest_pred = predictions[-1]  # 最後の行
            else:
                latest_pred = predictions

            # 予測結果を3クラス（down, range, up）の確率に変換
            if latest_pred.shape[0] == 3:
                return {
                    "down": float(latest_pred[0]),
                    "range": float(latest_pred[1]),
                    "up": float(latest_pred[2]),
                }
            elif latest_pred.shape[0] == 2:
                # 2クラス分類の場合、rangeを中間値として設定
                return {
                    "down": float(latest_pred[0]),
                    "range": 0.34,
                    "up": float(latest_pred[1]),
                }
            else:
                # 予期しない形式の場合、デフォルト値を返す
                logger.warning(f"予期しない予測結果の形式: {latest_pred.shape}")
                return self.config.prediction.get_default_predictions()

        except Exception as e:
            logger.warning(f"予測エラー: {e}")
            return self.config.prediction.get_default_predictions()  # デフォルト値

    def save_model(self, model_name: str = "ml_signal_model") -> str:
        """
        モデルを保存

        Args:
            model_name: モデル名

        Returns:
            保存パス
        """
        try:
            if not self.is_trained:
                raise MLModelError("学習済みモデルがありません")

            # ModelManagerを使用してモデルを保存
            metadata = {
                "model_type": "LightGBM",
                "feature_count": (
                    len(self.feature_columns) if self.feature_columns else 0
                ),
                "is_trained": self.is_trained,
            }

            model_path = model_manager.save_model(
                model=self.model,
                model_name=model_name,
                metadata=metadata,
                scaler=self.scaler,
                feature_columns=self.feature_columns,
            )

            if model_path is None:
                raise MLModelError("モデルの保存に失敗し、パスが返されませんでした。")

            logger.info(f"モデル保存完了: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            raise MLModelError(f"モデル保存に失敗しました: {e}")

    def load_model(self, model_path: str) -> bool:
        """
        モデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        try:
            # ModelManagerを使用してモデルを読み込み
            model_data = model_manager.load_model(model_path)

            if model_data is None:
                return False

            # モデルデータから各要素を取得
            self.model = model_data.get("model")
            self.scaler = model_data.get("scaler")
            self.feature_columns = model_data.get("feature_columns")

            if self.model is None:
                raise MLModelError("モデルデータにモデルが含まれていません")

            self.is_trained = True
            logger.info(f"モデル読み込み完了: {model_path}")
            return True

        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return False

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        try:
            if not self.is_trained or self.model is None:
                return {}

            if not self.feature_columns:
                return {}

            # 特徴量重要度（モデルタイプに応じて分岐）
            # LightGBMモデル
            importances = self.model.feature_importance(importance_type="gain")

            if not self.feature_columns:
                return {}

            importance = dict(zip(self.feature_columns, importances))

            # 重要度順にソート
            sorted_importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            )

            return sorted_importance

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return {}
