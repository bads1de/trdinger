"""
LightGBMモデルラッパー

アンサンブル学習で使用するLightGBMモデルのラッパークラスを提供します。
LightGBMTrainerの機能を簡略化してアンサンブル専用に最適化されています。
"""

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Optional

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class LightGBMModel:
    """
    アンサンブル内で使用するLightGBMモデルラッパー

    LightGBMTrainerの機能を簡略化してアンサンブル専用に最適化
    """

    def __init__(self, automl_config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            automl_config: AutoML設定（現在は未使用）
        """
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        self.automl_config = automl_config

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        LightGBMモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果
        """
        try:
            # 特徴量カラムを保存
            self.feature_columns = X_train.columns.tolist()

            # LightGBMデータセットを作成
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            # クラス数を判定
            num_classes = len(np.unique(y_train))

            # LightGBMパラメータ
            params = {
                "objective": "multiclass" if num_classes > 2 else "binary",
                "num_class": num_classes if num_classes > 2 else None,
                "metric": "multi_logloss" if num_classes > 2 else "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": 42,
            }

            # モデル学習
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                num_boost_round=100,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(0),  # ログを抑制
                ],
            )

            # 予測と評価
            y_pred_proba = self.model.predict(
                X_test, num_iteration=self.model.best_iteration
            )

            if num_classes > 2:
                y_pred_class = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred_class = (y_pred_proba > 0.5).astype(int)

            # 評価指標を計算
            from sklearn.metrics import accuracy_score

            accuracy = accuracy_score(y_test, y_pred_class)

            self.is_trained = True

            logger.info(f"LightGBM学習開始: {num_classes}クラス分類")
            logger.info(f"クラス分布: {dict(y_train.value_counts())}")
            logger.info(f"LightGBMモデル学習完了: 精度={accuracy:.4f}")

            return {
                "accuracy": accuracy,
                "num_classes": num_classes,
                "best_iteration": self.model.best_iteration,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

        except Exception as e:
            logger.error(f"LightGBMモデル学習エラー: {e}")
            raise UnifiedModelError(f"LightGBMモデル学習に失敗しました: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("学習済みモデルがありません")

        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        return self.predict(X)
