"""
LightGBM学習クラス

BaseMLTrainerを継承し、LightGBMを使用した具体的な学習実装を提供します。
"""

import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, cast
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from .base_ml_trainer import BaseMLTrainer
from ...utils.ml_error_handler import MLModelError

logger = logging.getLogger(__name__)


class LightGBMTrainer(BaseMLTrainer):
    """
    LightGBMを使用したML学習クラス

    BaseMLTrainerを継承し、LightGBM固有の学習ロジックを実装します。
    """

    def __init__(self):
        """初期化"""
        super().__init__()
        self.model_type = "LightGBM"

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測確率の配列 [下落確率, レンジ確率, 上昇確率]
        """
        if not self.is_trained or self.model is None:
            raise MLModelError("学習済みモデルがありません")

        try:
            # 特徴量を選択・前処理
            if self.feature_columns:
                # 学習時と同じ特徴量を選択
                available_features = [
                    col for col in self.feature_columns if col in features_df.columns
                ]
                if len(available_features) != len(self.feature_columns):
                    missing_features = set(self.feature_columns) - set(
                        available_features
                    )
                    logger.warning(f"不足している特徴量: {missing_features}")

                features_selected = features_df[available_features].fillna(0)
            else:
                # 数値列のみを選択
                numeric_columns = features_df.select_dtypes(include=[np.number]).columns
                features_selected = features_df[numeric_columns].fillna(0)

            # スケーリング
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_selected)
            else:
                features_scaled = features_selected.values

            # 予測実行
            predictions = cast(
                np.ndarray,
                self.model.predict(
                    features_scaled, num_iteration=self.model.best_iteration
                ),
            )

            # 3クラス分類の確率を返す
            if predictions.ndim == 1:
                # バイナリ分類の場合は3クラスに変換
                prob_up = predictions
                prob_down = 1 - predictions
                prob_range = np.zeros_like(predictions)
                return np.column_stack([prob_down, prob_range, prob_up])
            else:
                # マルチクラス分類の場合はそのまま返す
                return predictions

        except Exception as e:
            logger.error(f"予測エラー: {e}")
            # デフォルト値を返す（均等確率）
            return np.full((len(features_df), 3), 1 / 3)

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        LightGBMモデル学習の具体的な実装

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ラベル
            y_test: テスト用ラベル
            **training_params: 学習パラメータ

        Returns:
            学習結果
        """
        try:
            # LightGBMデータセットを作成
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            # LightGBMパラメータ（設定から取得）
            params = self.config.lightgbm.to_dict()

            # 追加パラメータを適用
            random_state = training_params.get("random_state", 42)
            params["random_state"] = random_state

            # クラス数を設定
            num_classes = len(np.unique(y_train))
            if num_classes > 2:
                params["objective"] = "multiclass"
                params["num_class"] = num_classes
                params["metric"] = "multi_logloss"
            else:
                params["objective"] = "binary"
                params["metric"] = "binary_logloss"

            logger.info(f"LightGBM学習開始: {num_classes}クラス分類")

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
                    lgb.log_evaluation(0),  # ログを抑制
                ],
            )

            # 予測と評価
            y_pred_proba = cast(
                np.ndarray,
                self.model.predict(X_test, num_iteration=self.model.best_iteration),
            )

            if num_classes > 2:
                y_pred_class = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred_class = (y_pred_proba > 0.5).astype(int)

            # 詳細な評価指標を計算
            detailed_metrics = self.calculate_detailed_metrics(
                y_test, y_pred_class, y_pred_proba
            )

            # 分類レポートも生成
            class_report = classification_report(
                y_test, y_pred_class, output_dict=True, zero_division=0.0
            )

            # 特徴量重要度
            feature_importance = {}
            if self.feature_columns:
                importances = self.model.feature_importance(importance_type="gain")
                feature_importance = dict(zip(self.feature_columns, importances))

            result = {
                **detailed_metrics,  # 詳細な評価指標を展開
                "classification_report": class_report,
                "feature_importance": feature_importance,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "best_iteration": self.model.best_iteration,
                "model_type": self.model_type,
                "num_classes": num_classes,
            }

            logger.info(f"LightGBMモデル学習完了: 精度={accuracy:.4f}")
            return result

        except Exception as e:
            logger.error(f"LightGBMモデル学習エラー: {e}")
            raise MLModelError(f"LightGBMモデル学習に失敗しました: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or self.model is None:
            raise MLModelError("学習済みモデルがありません")

        if not self.feature_columns:
            return {}

        importance_values = self.model.feature_importance(importance_type="gain")
        return dict(zip(self.feature_columns, importance_values))

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得

        Returns:
            モデル情報の辞書
        """
        if not self.is_trained or self.model is None:
            return {"is_trained": False}

        return {
            "is_trained": True,
            "model_type": self.model_type,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "best_iteration": getattr(self.model, "best_iteration", None),
            "feature_columns": self.feature_columns,
        }
