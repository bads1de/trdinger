"""
CatBoostモデルラッパー

アンサンブル学習で使用するCatBoostモデルのラッパークラスを提供します。
CatBoostを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class CatBoostModel:
    """
    アンサンブル内で使用するCatBoostモデルラッパー

    CatBoostを使用してアンサンブル専用に最適化されたモデル
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
    ) -> Dict[str, Any]:
        """
        CatBoostモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果
        """
        try:
            from catboost import CatBoostClassifier

            # 特徴量カラムを保存
            self.feature_columns = X_train.columns.tolist()

            # クラス数を判定
            num_classes = len(np.unique(y_train))

            # CatBoostパラメータ
            params = {
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 6,
                "l2_leaf_reg": 3,
                "bootstrap_type": "Bernoulli",  # Bayesian -> Bernoulli (subsampleと互換性あり)
                "subsample": 0.8,
                "random_seed": 42,
                "verbose": False,
                "allow_writing_files": False,  # ファイル出力を無効化
            }

            # 多クラス分類の場合の設定
            if num_classes > 2:
                params["loss_function"] = "MultiClass"
                params["classes_count"] = num_classes
            else:
                params["loss_function"] = "Logloss"

            # モデルを作成
            self.model = CatBoostClassifier(**params)

            # モデル学習
            self.model.fit(
                X_train,
                y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=50,
                verbose=False,
            )

            # 予測と評価
            y_pred_proba = self.model.predict_proba(X_test)

            if num_classes > 2:
                y_pred_class = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred_class = (y_pred_proba[:, 1] > 0.5).astype(int)

            # 評価指標を計算
            from sklearn.metrics import accuracy_score

            accuracy = accuracy_score(y_test, y_pred_class)

            self.is_trained = True

            logger.info(f"CatBoost学習開始: {num_classes}クラス分類")
            logger.info(f"クラス分布: {dict(y_train.value_counts())}")
            logger.info(f"CatBoostモデル学習完了: 精度={accuracy:.4f}")

            return {
                "accuracy": accuracy,
                "num_classes": num_classes,
                "best_iteration": self.model.get_best_iteration(),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

        except ImportError:
            logger.error(
                "CatBoostがインストールされていません。pip install catboostを実行してください。"
            )
            raise UnifiedModelError("CatBoostがインストールされていません")
        except Exception as e:
            logger.error(f"CatBoostモデル学習エラー: {e}")
            raise UnifiedModelError(f"CatBoostモデル学習に失敗しました: {e}")

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

        try:
            predictions = self.model.predict_proba(X)
            return predictions
        except Exception as e:
            raise UnifiedModelError(f"CatBoost予測エラー: {e}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        return self.predict(X)
