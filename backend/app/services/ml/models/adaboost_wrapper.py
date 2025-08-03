"""
AdaBoostモデルラッパー

アンサンブル学習で使用するAdaBoostモデルのラッパークラスを提供します。
scikit-learnのAdaBoostClassifierを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class AdaBoostModel:
    """
    アンサンブル内で使用するAdaBoostモデルラッパー

    scikit-learnのAdaBoostClassifierを使用してアンサンブル専用に最適化されたモデル
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

        # ベース推定器
        self.base_estimator = DecisionTreeClassifier(
            max_depth=3, class_weight="balanced", random_state=42
        )

        # デフォルトパラメータ
        self.default_params = {
            "estimator": self.base_estimator,
            "n_estimators": 100,
            "learning_rate": 1.0,
            "algorithm": "SAMME.R",
            "random_state": 42,
        }

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        AdaBoostモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果の辞書
        """
        try:
            logger.info("🚀 AdaBoostモデルの学習を開始")

            # 特徴量カラムを保存
            self.feature_columns = list(X_train.columns)

            # モデル初期化
            self.model = AdaBoostClassifier(**self.default_params)

            # 学習実行
            self.model.fit(X_train, y_train)

            # 予測
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            # 評価指標計算
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_balanced_acc = balanced_accuracy_score(y_train, y_pred_train)
            test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
            train_f1 = f1_score(y_train, y_pred_train, average="weighted")
            test_f1 = f1_score(y_test, y_pred_test, average="weighted")

            # 特徴量重要度
            feature_importance = dict(
                zip(self.feature_columns, self.model.feature_importances_)
            )

            self.is_trained = True

            results = {
                "algorithm": "adaboost",
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_balanced_accuracy": train_balanced_acc,
                "test_balanced_accuracy": test_balanced_acc,
                "train_f1_score": train_f1,
                "test_f1_score": test_f1,
                "feature_importance": feature_importance,
                "n_estimators": self.model.n_estimators,
                "learning_rate": self.model.learning_rate,
                "algorithm": self.model.algorithm,
                "feature_count": len(self.feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

            logger.info(f"✅ AdaBoost学習完了 - テスト精度: {test_accuracy:.4f}")
            return results

        except Exception as e:
            logger.error(f"❌ AdaBoost学習エラー: {e}")
            raise UnifiedModelError(f"AdaBoost学習に失敗しました: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        try:
            # 特徴量の順序を確認
            if self.feature_columns:
                X = X[self.feature_columns]

            predictions = self.model.predict(X)
            return predictions

        except Exception as e:
            logger.error(f"AdaBoost予測エラー: {e}")
            raise UnifiedModelError(f"予測に失敗しました: {e}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率の配列
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        try:
            # 特徴量の順序を確認
            if self.feature_columns:
                X = X[self.feature_columns]

            probabilities = self.model.predict_proba(X)
            return probabilities

        except Exception as e:
            logger.error(f"AdaBoost確率予測エラー: {e}")
            raise UnifiedModelError(f"確率予測に失敗しました: {e}")

    @property
    def feature_columns(self) -> List[str]:
        """特徴量カラム名のリストを取得"""
        return self._feature_columns

    @feature_columns.setter
    def feature_columns(self, columns: List[str]):
        """特徴量カラム名のリストを設定"""
        self._feature_columns = columns

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        if not self.feature_columns:
            raise UnifiedModelError("特徴量カラムが設定されていません")

        return dict(zip(self.feature_columns, self.model.feature_importances_))

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得

        Returns:
            モデル情報の辞書
        """
        if not self.is_trained or self.model is None:
            return {"status": "not_trained"}

        return {
            "algorithm": "adaboost",
            "n_estimators": self.model.n_estimators,
            "learning_rate": self.model.learning_rate,
            "algorithm_type": self.model.algorithm,
            "base_estimator_max_depth": self.base_estimator.max_depth,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "status": "trained",
        }
